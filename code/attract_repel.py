import os
import numpy
import time
import random
import math
from numpy.linalg import norm
from numpy import dot
import numpy as np
import codecs
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def FloatTensorWrapper(tensor, cuda=0):
    if cuda >= 0:
        tensor = torch.FloatTensor(tensor).cuda(cuda)
    else:
        tensor = torch.FloatTensor(tensor)
    return tensor


def LongTensorWrapper(tensor, cuda=0):
    if cuda >= 0:
        tensor = torch.LongTensor(tensor).cuda(cuda)
    else:
        tensor = torch.LongTensor(tensor)
    return tensor


def l2_loss(input_tensor, target_tensor):
    loss_matrix = nn.functional.mse_loss(input_tensor, target_tensor, reduction="none")
    return torch.sum(loss_matrix)/2


class PytorchModel(torch.nn.Module):
    def __init__(self, W, attract_margin_value=1.0, repel_margin_value=0.0, l2_reg_constant=1e-9):
        super(PytorchModel, self).__init__()
        self.attract_margin = attract_margin_value
        self.repel_margin = repel_margin_value
        self.regularisation_constant = l2_reg_constant
        self.init_W = nn.Embedding(W.shape[0], W.shape[1])
        self.init_W.weight = nn.Parameter(torch.FloatTensor(W), requires_grad=False)
        self.dynamic_W = nn.Embedding(W.shape[0], W.shape[1])
        self.dynamic_W.weight = nn.Parameter(torch.FloatTensor(W), requires_grad=True)

    def attract_cost(self, attract_examples, negative_examples_attract, trainable_embs):
        np_attract_examples = np.array(attract_examples)
        np_negative_examples_attract = np.array(negative_examples_attract)

        attract_examples_left = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_attract_examples[:, 0]))))
        attract_examples_right = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_attract_examples[:, 1]))))

        negative_examples_attract_left = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_negative_examples_attract[:, 0]))))
        negative_examples_attract_right = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_negative_examples_attract[:, 1]))))

        # dot product between the example pairs.
        attract_similarity_between_examples = torch.sum(torch.mul(attract_examples_left, attract_examples_right), 1)

        # dot product of each word in the example with its negative example.
        attract_similarity_to_negatives_left = torch.sum(torch.mul(attract_examples_left, negative_examples_attract_left), 1)
        attract_similarity_to_negatives_right = torch.sum(torch.mul(attract_examples_right, negative_examples_attract_right), 1)

        attract_cost = nn.functional.relu(self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
            nn.functional.relu(self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

        original_attract_examples_left = self.init_W(LongTensorWrapper(np_attract_examples[:, 0]))
        original_attract_examples_right = self.init_W(LongTensorWrapper(np_attract_examples[:, 1]))

        # and then define the respective regularisation costs:
        regularisation_cost_attract = self.regularisation_constant * (l2_loss(original_attract_examples_left, attract_examples_left) + l2_loss(original_attract_examples_right, attract_examples_right))

        attract_cost += regularisation_cost_attract

        return attract_cost

    def repel_cost(self, repel_examples, negative_examples_repel, trainable_embs):
        np_repel_examples = np.array(repel_examples)
        np_negative_examples_repel = np.array(negative_examples_repel)

        repel_examples_left = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_repel_examples[:, 0]))))
        repel_examples_right = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_repel_examples[:, 1]))))

        negative_examples_repel_left = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_negative_examples_repel[:, 0]))))
        negative_examples_repel_right = nn.functional.normalize(trainable_embs(Variable(LongTensorWrapper(np_negative_examples_repel[:, 1]))))

        # dot product between the example pairs.
        repel_similarity_between_examples = torch.sum(torch.mul(repel_examples_left, repel_examples_right), 1)

        # dot product of each word in the example with its negative example.
        repel_similarity_to_negatives_left = torch.sum(torch.mul(repel_examples_left, negative_examples_repel_left), 1)
        repel_similarity_to_negatives_right = torch.sum(torch.mul(repel_examples_right, negative_examples_repel_right), 1)

        repel_cost = nn.functional.relu(self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
            nn.functional.relu(self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)

        # load the original distributional vectors for the example pairs:
        original_repel_examples_left = self.init_W(LongTensorWrapper(np_repel_examples[:, 0]))
        original_repel_examples_right = self.init_W(LongTensorWrapper(np_repel_examples[:, 1]))

        # and then define the respective regularisation costs:
        regularisation_cost_repel = self.regularisation_constant * (l2_loss(original_repel_examples_left, repel_examples_left) + l2_loss(original_repel_examples_right, repel_examples_right))
        repel_cost += regularisation_cost_repel
        return repel_cost


class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for an Attract-Repel run.
    """

    def __init__(self, params, is_target=False):
        """
        To initialise the class, we need to supply the params, which contains the location of
        the pretrained (distributional) word vectors, the location of (potentially more than one)
        collections of linguistic constraints (one pair per line), as well as the
        hyperparameters of the Attract-Repel procedure.
        """
        self.params = params

        if is_target:
            distributional_vectors_filepath = self.params.target_vectors
        else:
            distributional_vectors_filepath = self.params.distributional_vectors
        self.output_filepath = os.path.join(self.params.out_dir, "ar_vectors")

        # load initial distributional word vectors.
        distributional_vectors = load_word_vectors(distributional_vectors_filepath, max_vocab=200000)
        print("Scores (Spearman's rho coefficient) of initial vectors are:\n")
        self.language = 'english' if not is_target else self.params.target_lang
        self.simlex_scores(distributional_vectors, language=self.language, no_simlex=self.params.no_simlex)

        self.vocabulary = set(distributional_vectors.keys())

        # this will be used to load constraints
        self.vocab_index = {}
        self.inverted_index = {}

        for idx, word in enumerate(self.vocabulary):
            self.vocab_index[word] = idx
            self.inverted_index[idx] = word

        self.synonyms = set()
        self.antonyms = set()

        # load list of filenames for synonyms and antonyms.
        if not is_target:
            synonym_list = self.params.attract_constraints
            antonym_list = self.params.repel_constraints
        else:
            synonym_list = self.params.attract_tgtctr
            antonym_list = self.params.repel_tgtctr

        synonym_list = [synonym_list] if type(synonym_list) == str else synonym_list
        antonym_list = [antonym_list] if type(antonym_list) == str else antonym_list

        if synonym_list:
            for syn_filepath in synonym_list:
                self.synonyms = self.synonyms | self.load_constraints(syn_filepath)

        if antonym_list:
            for ant_filepath in antonym_list:
                self.antonyms = self.antonyms | self.load_constraints(ant_filepath)

        # finally, load the experiment hyperparameters:
        self.load_experiment_hyperparameters()

        self.embedding_size = random.choice(list(distributional_vectors.values())).shape[0]
        self.vocabulary_size = len(self.vocabulary)

        # Next, prepare the matrix of initial vectors and initialise the model.
        numpy_embedding = numpy.zeros((self.vocabulary_size, self.embedding_size), dtype="float32")
        for idx in range(0, self.vocabulary_size):
            numpy_embedding[idx, :] = distributional_vectors[self.inverted_index[idx]]

        self.model = PytorchModel(numpy_embedding,
                                  attract_margin_value=self.attract_margin_value,
                                  repel_margin_value=self.repel_margin_value,
                                  l2_reg_constant=self.regularisation_constant_value
                                  )
        if params.cuda:
            self.model.cuda()

    def add_new_constraints(self, attract_constraints, repel_constraints):
        self.model.dynamic_W.weight.data.copy_(self.model.init_W.weight.data)
        self.synonyms = set()
        self.antonyms = set()

        for word_pair in attract_constraints:
            if word_pair[0] in self.vocabulary and word_pair[1] in self.vocabulary and word_pair[0] != word_pair[1]:
                self.synonyms |= {(self.vocab_index[word_pair[0]], self.vocab_index[word_pair[1]])}

        for word_pair in repel_constraints:
            if word_pair[0] in self.vocabulary and word_pair[1] in self.vocabulary and word_pair[0] != word_pair[1]:
                self.antonyms |= {(self.vocab_index[word_pair[0]], self.vocab_index[word_pair[1]])}

    def load_constraints(self, constraints_filepath):
        """
        This methods reads a collection of constraints from the specified file, and returns a set with
        all constraints for which both of their constituent words are in the specified vocabulary.
        """
        constraints_filepath.strip()
        constraints = set()
        with codecs.open(constraints_filepath, "r", "utf-8") as f:
            for line in f:
                word_pair = line.split()
                if word_pair[0] in self.vocabulary and word_pair[1] in self.vocabulary and word_pair[0] != word_pair[1]:
                    constraints |= {(self.vocab_index[word_pair[0]], self.vocab_index[word_pair[1]])}
        return constraints

    def load_experiment_hyperparameters(self):
        """
        This method loads/sets the hyperparameters of the procedure as specified in the paper.
        """
        self.attract_margin_value = self.params.attract_margin
        self.repel_margin_value = self.params.repel_margin
        self.batch_size = self.params.batch_size_ar
        self.regularisation_constant_value = self.params.l2_reg_constant
        self.max_iter = self.params.max_iter
        self.no_simlex = self.params.no_simlex

        print("\nExperiment hyperparameters (attract_margin, repel_margin, batch_size, l2_reg_constant, max_iter):",
              self.attract_margin_value, self.repel_margin_value, self.batch_size, self.regularisation_constant_value, self.max_iter)

    def extract_negative_examples(self, list_minibatch, attract_batch=True, from_emb=None):
        """
        For each example in the minibatch, this method returns the closest vector which is not
        in each words example pair.
        """
        from_emb = self.model.dynamic_W if from_emb is None else from_emb
        np_list_minibatch = np.array(list_minibatch)

        list_of_representations = []
        list_of_indices = []
        lefts = Variable(LongTensorWrapper(np_list_minibatch[:, 0]))
        rights = Variable(LongTensorWrapper(np_list_minibatch[:, 1]))
        representations = [nn.functional.normalize(from_emb(lefts)).data.cpu().numpy(), nn.functional.normalize(from_emb(rights)).data.cpu().numpy()]

        for idx, (example_left, example_right) in enumerate(list_minibatch):

            list_of_representations.append(representations[0][idx])
            list_of_representations.append(representations[1][idx])

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine')
        square_distance_list = squareform(condensed_distance_list)

        if attract_batch:
            default_value = 2.0  # value to set for given attract/repel pair, so that it can not be found as closest or furthest away.
        else:
            default_value = 0.0  # for antonyms, we want the opposite value from the synonym one. Cosine Distance is [0,2].

        for i in range(len(square_distance_list)):

            square_distance_list[i, i] = default_value

            if i % 2 == 0:
                square_distance_list[i, i+1] = default_value
            else:
                square_distance_list[i, i-1] = default_value

        if attract_batch:
            negative_example_indices = numpy.argmin(square_distance_list, axis=1)  # for each of the 100 elements, finds the index which has the minimal cosine distance (i.e. most similar).
        else:
            negative_example_indices = numpy.argmax(square_distance_list, axis=1)  # for antonyms, find the least similar one.

        negative_examples = []

        for idx in range(len(list_minibatch)):

            negative_example_left = list_of_indices[negative_example_indices[2 * idx]]
            negative_example_right = list_of_indices[negative_example_indices[2 * idx + 1]]

            negative_examples.append((negative_example_left, negative_example_right))

        negative_examples = mix_sampling(list_minibatch, negative_examples)

        return negative_examples

    def attract_repel(self):
        """
        This method repeatedly applies optimisation steps to fit the word vectors to the provided linguistic constraints.
        """

        current_iteration = 0

        self.model.dynamic_W.weight.requires_grad = True

        # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
        for antonym_pair in self.antonyms:
            if antonym_pair in self.synonyms:
                self.synonyms.remove(antonym_pair)

        self.synonyms = list(self.synonyms)
        self.antonyms = list(self.antonyms)

        self.syn_count = len(self.synonyms)
        self.ant_count = len(self.antonyms)

        print("\nAntonym pairs:", len(self.antonyms), "Synonym pairs:", len(self.synonyms))

        syn_batches = int(self.syn_count / self.batch_size)
        ant_batches = int(self.ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print("\nRunning the optimisation procedure for", self.max_iter, "iterations...")

        last_time = time.time()

        # set optimizer
        attract_optimizer = torch.optim.Adagrad(self.model.dynamic_W.parameters(), lr=0.05, initial_accumulator_value=0.1)
        repel_optimizer = torch.optim.Adagrad(self.model.dynamic_W.parameters(), lr=0.05, initial_accumulator_value=0.1)
        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_counter = 0
            synonym_counter = 0

            order_of_synonyms = [i for i in range(0, self.syn_count)]
            order_of_antonyms = [i for i in range(0, self.ant_count)]

            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym batch
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:] = [1] * ant_batches  # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print("Starting epoch:", current_iteration+1)
            else:
                print("Starting epoch:", current_iteration+1, "Last epoch took:", round(time.time() - last_time, 1), "seconds.")
                last_time = time.time()

            for batch_index in range(0, batches_per_epoch):

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:
                    synonymy_examples = [self.synonyms[order_of_synonyms[x]] for x in range(synonym_counter * self.batch_size, (synonym_counter+1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(synonymy_examples, attract_batch=True)

                    attract_cost = self.model.attract_cost(synonymy_examples, current_negatives, self.model.dynamic_W)
                    # apply gradients
                    self.model.zero_grad()
                    torch.sum(attract_cost).backward()
                    self.model.dynamic_W.weight.grad.data.clamp_(-2.0, 2.0)
                    attract_optimizer.step()
                    synonym_counter += 1

                else:

                    antonymy_examples = [self.antonyms[order_of_antonyms[x]] for x in range(antonym_counter * self.batch_size, (antonym_counter+1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(antonymy_examples, attract_batch=False)
                    repel_cost = self.model.repel_cost(antonymy_examples, current_negatives, self.model.dynamic_W)
                    # apply gradients
                    self.model.zero_grad()
                    torch.sum(repel_cost).backward()
                    self.model.dynamic_W.weight.grad.data.clamp_(-2.0, 2.0)
                    repel_optimizer.step()
                    antonym_counter += 1

            current_iteration += 1
            self.create_vector_dictionary()  # whether to print SimLex score at the end of each epoch

    def create_vector_dictionary(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if no_simlex=False) prints their SimLex scores.
        """

        current_vectors = self.model.dynamic_W.weight.data.cpu().numpy()
        self.word_vectors = {}
        for idx in range(0, self.vocabulary_size):
            self.word_vectors[self.inverted_index[idx]] = normalise_vector(current_vectors[idx, :])

        if not self.params.no_simlex:
            (score_simlex, score_wordsim) = self.simlex_scores(self.word_vectors, language=self.language)
            return (score_simlex, score_wordsim)

        return (1.0, 1.0)

    def print_word_vectors(self, n_b):
        """
        This function prints the collection of word vectors to file, in a plain textual format.
        """
        f_name = self.output_filepath + str(n_b) + ".txt"
        f_write = codecs.open(f_name, 'w', 'utf-8')

        for key in self.word_vectors:
            f_write.write(key+" "+" ".join(map(str, self.word_vectors[key]))+"\n")

        print("Printed", len(self.word_vectors), "word vectors to:", f_name)

    def simlex_scores(self, word_vectors, language="english", no_simlex=False):
        simverb_score = 0
        simlex_score, simlex_coverage = simlex_analysis(word_vectors, language, "simlex", no_simlex)
        print("SimLex score for", language, "is:", simlex_score, "coverage:", simlex_coverage, "/ 999")

        if language == "english":
            simverb_score, simverb_coverage = simlex_analysis(word_vectors, language, source="simverb")
            print("SimVerb score for english is:", simverb_score, "coverage:", simverb_coverage, "/ 3500")

        # ws_score, ws_coverage = simlex_analysis(word_vectors, language, source="wordsim")
        # print("WordSim score for", "english", "is:", ws_score, "coverage:", ws_coverage, "/ 353\n")

        return simlex_score, simverb_score


def simlex_analysis(word_vectors, language="english", source="simlex", no_simlex=False):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    if no_simlex:
        return (0.0, 0)
    lang2code = defaultdict(str)
    code = lang2code[language]
    pair_list = []
    if source == "simlex":
        fread_simlex = codecs.open("../evaluation/simlex-" + language + ".txt", 'r', 'utf-8')
    elif source == "simverb":
        fread_simlex = codecs.open("../evaluation/simverb.txt", 'r', 'utf-8')
    elif source == "wordsim":
        fread_simlex = codecs.open("../evaluation/ws-353/wordsim353-" + language + ".txt", 'r', 'utf-8')  # specify english, english-rel, etc.

    line_number = 0
    for line in fread_simlex:

        if line_number > 0:

            tokens = line.split()
            word_i = code + tokens[0].lower()
            word_j = code + tokens[1].lower()
            score = float(tokens[2])

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append(((word_i, word_j), score))
            else:
                pass

        line_number += 1

    if not pair_list:
        return (0.0, 0)

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:

        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return round(spearman_rho[0], 3), coverage


def normalise_vector(v1):
    return v1 / norm(v1)


def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def random_different_from(top_range, number_to_not_repeat):

    result = random.randint(0, top_range-1)
    while result == number_to_not_repeat:
        result = random.randint(0, top_range-1)

    return result


def mix_sampling(list_of_examples, negative_examples):
    """
    Converts half of the negative examples to random words from the batch (that are not in the given example pair).
    """
    mixed_negative_examples = []
    batch_size = len(list_of_examples)

    for idx, (left_idx, right_idx) in enumerate(negative_examples):

        new_left = left_idx
        new_right = right_idx

        if random.random() >= 0.5:
            new_left = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]

        if random.random() >= 0.5:
            new_right = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]

        mixed_negative_examples.append((new_left, new_right))

    return mixed_negative_examples


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination, max_vocab=50000):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print "Loading pretrained word vectors from", file_destination
    word_dictionary = {}

    try:
        f = codecs.open(file_destination, 'r', 'utf-8')
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            if i == 0:
                if len(line.split()) == 2:
                    continue
            line = line.split(" ", 1)
            key = unicode(line[0].lower())
            word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")

    except IOError:
        print "Word vectors could not be loaded from:", file_destination
        return {}

    print len(word_dictionary), "vectors loaded from", file_destination

    return word_dictionary
