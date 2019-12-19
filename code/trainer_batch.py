import os
import re
import logging
import numpy as np
import inspect
import codecs
import time
import json
from collections import OrderedDict
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from attract_repel import simlex_analysis


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class MaxMargin_Loss(torch.nn.Module):

    def __init__(self, params):
        super(MaxMargin_Loss, self).__init__()
        self.params = params

    def forward(self, y_pred, y_true):
        cost = 0.
        for i in xrange(0, self.params.sim_neg):
            new_true = torch.randperm(self.params.batch_size)
            new_true = new_true.cuda() if self.params.cuda else new_true
            new_true = y_true[new_true]
            mg = self.params.sim_margin - F.cosine_similarity(y_true, y_pred) + F.cosine_similarity(new_true, y_pred)
            cost += torch.clamp(mg, min=0)
        return cost.mean()


class Trainer(object):

    def __init__(self, ctr_idx, dico, ar, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = ar.model.init_W
        self.tgt_emb = ar.model.dynamic_W
        self.mapping = mapping
        self.discriminator = discriminator
        self.train_idx = ctr_idx
        self.dico = dico
        self.params = params
        self.language = ar.language
        # optimizers
        if hasattr(params, 'map_optimizer') and params.adversarial:
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer') and params.adversarial:
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        if hasattr(params, 'sim_optimizer'):
            optim_fn, optim_params = get_optimizer(params.sim_optimizer)
            self.sim_optimizer = optim_fn(mapping.parameters(), **optim_params)
        self.max_margin = MaxMargin_Loss(params)
        # best validation score
        self.best_valid_metric = 0
        self.adversarial = params.adversarial

        self.decrease_lr = False

    def run_experiment(self, params, evaluator, afx=""):
        self.patience = 0
        for n_epoch in range(params.n_epochs):
            logging.info('Starting postspec training epoch %i...' % n_epoch)
            tic = time.time()
            n_words_proc = 0
            stats = {'DIS_COSTS': [], 'GEN_COSTS': [], 'SIM_COSTS': []}
            for batch_index in range(0, params.epoch_size, params.batch_size):

                if params.adversarial:
                    # discriminator training
                    for _ in range(params.dis_steps):
                        self.dis_step(stats)
                    # mapping training (discriminator fooling)
                    self.mapping_step(stats, params)

                # similarity training
                n_words_proc += self.sim_step(stats)

                # log stats
                if batch_index % 500 == 0:
                    stats_str = [('DIS_COSTS', 'Discriminator loss'),
                                 ('GEN_COSTS', 'Generator loss'),
                                 ('SIM_COSTS', 'Similarity loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                 for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                    logging.info(('%06i - ' % batch_index) + ' - '.join(stats_log))

                    # reset
                    tic = time.time()
                    n_words_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]

            # embeddings / discriminator evaluation
            to_log = OrderedDict({'n_epoch': n_epoch})
            evaluator.all_eval(to_log)

            VALIDATION_METRIC = 'mean_cosine'

            # JSON log / save best model / end of epoch
            logging.info("__log__:%s" % json.dumps(to_log))
            self.save_best(to_log, VALIDATION_METRIC, afx)
            logging.info('End of epoch %i.\n\n' % n_epoch)

            if self.patience >= self.params.patience:
                print("Reached max patience")
                break

    def get_sim_xy(self, examples):
        """
        Get similarity input batch / output target.
        """
        # tensorfy IDs
        ids = torch.LongTensor(examples)
        if self.params.cuda:
            ids = ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(ids))
        tgt_emb = self.tgt_emb(Variable(ids))
        src_emb = self.mapping(Variable(src_emb.data))
        tgt_emb = Variable(tgt_emb.data)

        return src_emb, tgt_emb

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        src_ids = np.random.choice(self.train_idx, size=bs)
        src_ids = torch.LongTensor(src_ids)
        tgt_ids = np.random.choice(self.train_idx, size=bs)
        tgt_ids = torch.LongTensor(tgt_ids)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        yp = torch.FloatTensor([self.params.dis_smooth] * bs)
        yn = torch.FloatTensor([1 - self.params.dis_smooth] * bs)
        yp = Variable(yp.cuda() if self.params.cuda else yp)
        yn = Variable(yn.cuda() if self.params.cuda else yn)

        return src_emb, tgt_emb, yp, yn

    def sim_step(self, stats):
        """
        Train the similarity between mapped src and tgt
        """
        if self.discriminator:
            self.discriminator.eval()
        # loss
        ids = random.sample(self.train_idx, self.params.batch_size)
        x, y = self.get_sim_xy(ids)
        ycos = torch.Tensor([1.] * self.params.batch_size)
        ycos = ycos.cuda() if self.params.cuda else ycos
        if self.params.sim_loss == "mse":
            loss = F.cosine_embedding_loss(x, y, Variable(ycos))
        elif self.params.sim_loss == "max_margin":
            loss = self.max_margin(x, y)
        else:
            raise Exception('Unknown similarity loss: "%s"' % self.params.sim_loss)
        loss = self.params.sim_lambda * loss
        stats['SIM_COSTS'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logging.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.sim_optimizer.zero_grad()
        loss.backward()
        self.sim_optimizer.step()

        return 2 * self.params.batch_size

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        xp, xn, yp, yn = self.get_dis_xy(volatile=True)
        for x, y in [(xp, yp), (xn, yn)]:
            preds = self.discriminator(Variable(x.data))
            loss = self.params.dis_lambda * F.binary_cross_entropy(preds, y)
            stats['DIS_COSTS'].append(loss.data[0])

            # check NaN
            if (loss != loss).data.any():
                logging.error("NaN detected (discriminator)")
                exit()

            # optim
            self.dis_optimizer.zero_grad()
            loss.backward()
            self.dis_optimizer.step()
            clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats, params):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        xp, xn, yp, yn = self.get_dis_xy(volatile=False)
        x = torch.cat([xp, xn], 0)
        y = torch.cat([yp, yn], 0)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.gen_lambda * loss
        stats['GEN_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logging.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.sim_optimizer:
            return
        old_lr = self.sim_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logging.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.sim_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1:
            if to_log[metric] < self.best_valid_metric:
                logging.info("Validation metric is lower than the best: %.5f vs %.5f".format(
                             to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.sim_optimizer.param_groups[0]['lr']
                    self.sim_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logging.info("Shrinking the learning rate: %.5f -> %.5f".format(
                                old_lr, self.sim_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric, afx):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logging.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            path = os.path.join(self.params.out_dir, 'best_mapping' + afx + '.t7')
            checkpoint = {'mapping': self.mapping.state_dict()}
            if self.discriminator:
                checkpoint['discriminator'] = self.discriminator.state_dict()
            logging.info('* Saving the mapping parameters to %s ...' % path)
            torch.save(checkpoint, path)
            self.patience = 0
        else:
            self.patience += 1

    def reload_best(self, afx=""):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.out_dir, 'best_mapping' + afx + '.t7')
        logging.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        checkpoint = torch.load(path)
        self.mapping.load_state_dict(checkpoint['mapping'])
        if self.discriminator:
            self.discriminator.load_state_dict(checkpoint['discriminator'])

    def heldoutall(self, params, out_dico, out_emb, afx=""):
        logging.info("Exporting mapped embeddings...")
        self.mapping.eval()
        with torch.no_grad():
            mapped_emb = self.mapping(Variable(out_emb)).data.cpu().numpy()
        ar_emb = self.tgt_emb.weight.data.cpu().numpy()

        # Now translate the unseen words to the target AR-specialised vector space
        all_keys = out_dico.word2id.keys()
        fhel = codecs.open(params.out_dir + "gold_embs" + afx + ".txt", "w")
        fall = codecs.open(params.out_dir + "silver_embs" + afx + ".txt", "w")
        for key in all_keys:
            hv = mapped_emb[out_dico.index(key)]
            hv = hv / np.linalg.norm(hv)
            hv = map(str, list(hv))
            hv = " ".join([str(key)] + hv) + "\n"
            fhel.write(hv)
            if key in self.dico.word2id:
                av = ar_emb[self.dico.index(key)]
                av = av / np.linalg.norm(av)
                av = map(str, list(av))
                av = " ".join([str(key)] + av) + "\n"
                fall.write(av)
            else:
                fall.write(hv)

        fhel.close()
        fall.close()
        logging.info("...Done!")

    def xling(self, params, ar_tgt, afx=""):
        emb = ar_tgt.model.init_W.weight.data
        dico = ar_tgt.inverted_index
        logging.info("Exporting mapped embeddings...")
        self.mapping.eval()
        with torch.no_grad():
            mapped_emb = self.mapping(emb).data.cpu().numpy()

        keyed_embs = {}
        fhel = codecs.open(params.out_dir + afx + ".txt", "w", encoding="utf8")
        for i, hv in enumerate(mapped_emb):
            word = dico[i]
            hv = hv / np.linalg.norm(hv)
            keyed_embs[word] = hv
            hv = map(str, list(hv))
            hv = " ".join([word] + hv) + "\n"
            fhel.write(hv)
        fhel.close()
        if not params.no_simlex:
            simlex_score, simlex_coverage = simlex_analysis(keyed_embs, ar_tgt.language)
            logging.info("SimLex score for %s is %s coverage: %s / 999", ar_tgt.language, str(simlex_score), str(simlex_coverage))
        logging.info("...Done!")

    def xling_mixed(self, ctr_idx, params, ar_tgt, afx=""):
        ctr_idx = set(ctr_idx)
        emb = ar_tgt.model.init_W.weight.data
        dico = ar_tgt.inverted_index
        logging.info("Exporting mapped embeddings...")
        self.mapping.eval()
        with torch.no_grad():
            mapped_emb = self.mapping(emb).data.cpu().numpy()
        ar_emb = ar_tgt.model.dynamic_W.weight.data.cpu().numpy()

        keyed_embs = {}
        fhel = codecs.open(params.out_dir + afx + ".txt", "w", encoding="utf8")
        for i in range(len(dico)):
            word = dico[i]
            if i in ctr_idx:
                hv = ar_emb[i]
            else:
                hv = mapped_emb[i]
            hv = hv / np.linalg.norm(hv)
            keyed_embs[word] = hv
            hv = map(str, list(hv))
            hv = " ".join([word] + hv) + "\n"
            fhel.write(hv)
        fhel.close()
        if not params.no_simlex:
            simlex_score, simlex_coverage = simlex_analysis(keyed_embs, ar_tgt.language)
            logging.info("SimLex score for %s is %s coverage: %s / 999", ar_tgt.language, str(simlex_score), str(simlex_coverage))
        logging.info("...Done!")
