#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division


import random
import time
import codecs
from collections import defaultdict
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.autograd import Variable

from attract_repel import LongTensorWrapper, FloatTensorWrapper, random_different_from
from ctr_pred_models import STM
from trainer_batch import get_optimizer


class CtrPredictor(object):

    def __init__(self, trainer, ar_src, ar_tgt, dis, label_smoothing_epsilon=0.1):
        """
        Initialize new constraint predictor.
        """
        super(CtrPredictor, self).__init__()
        self.vocab_size_src = len(ar_src.inverted_index)
        self.vocab_size_tgt = len(ar_tgt.inverted_index)
        self.params = trainer.params
        self.attract_constraints = ar_src.synonyms
        self.repel_constraints = ar_src.antonyms

        self.attract_ctr_tgt = list(ar_tgt.synonyms)
        self.repel_ctr_tgt = list(ar_tgt.antonyms)

        self.mapping = trainer.mapping
        self.source_index = ar_src.inverted_index
        self.inverted_index = ar_tgt.inverted_index
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.create_mapping(ar_src.model.init_W, ar_tgt.model.init_W)

        self.best_validation = 0
        self.patience = 0
        self.confidence_level_syn = 0.0
        self.confidence_level_ant = 0.0

    def routine(self, refine=False):
        self.create_triples()

        new_pairs_idx = []
        for relnum, synsants in enumerate([self.attract_ctr_tgt, self.repel_ctr_tgt]):
            self.STM = STM(self.map_emb, self.map_tgt, st_hid_dim=300, K=5, dropout=0.5)
            print(self.STM.parameters())
            if self.params.cuda:
                self.STM.cuda()
            self.stm_train(relnum)
            self.STM.eval()
            with torch.no_grad():
                triples = self.refine_candidates(synsants)
            new_pairs_idx.append(triples)
        new_pairs_idx = self.get_words(new_pairs_idx)
        return new_pairs_idx

    def get_words(self, idxs):
        new_syns, new_ants = idxs
        new_syns = [(self.inverted_index[l], self.inverted_index[r]) for l, r in new_syns]
        new_ants = [(self.inverted_index[l], self.inverted_index[r]) for l, r in new_ants]
        return new_syns, new_ants

    def create_mapping(self, emb, emb_tgt):
        mapping = self.mapping
        mapping.eval()

        with torch.no_grad():
            mapped = emb.weight.data
        self.map_emb = nn.Embedding(mapped.shape[0], mapped.shape[1])
        self.map_emb.weight = nn.Parameter(mapped, requires_grad=False)

        with torch.no_grad():
            mapped_tgt = emb_tgt.weight.data
        self.map_tgt = nn.Embedding(mapped_tgt.shape[0], mapped_tgt.shape[1])
        self.map_tgt.weight = nn.Parameter(mapped_tgt, requires_grad=False)

    def create_triples(self):

        attract_stm = list(self.attract_constraints)
        repel_stm = list(self.repel_constraints)

        lim_att_stm = int(len(attract_stm) * .95)
        lim_rep_stm = int(len(repel_stm) * .95)

        train_triples_stm = []
        dev_triples_stm = []

        for i, (l, r) in enumerate(attract_stm):
            if i < lim_att_stm:
                train_triples_stm.append((r, 0, l))
                train_triples_stm.append((l, 0, r))
            else:
                dev_triples_stm.append((r, 0, l))
                dev_triples_stm.append((l, 0, r))

        for i, (l, r) in enumerate(repel_stm):
            if i < lim_rep_stm:
                train_triples_stm.append((r, 1, l))
                train_triples_stm.append((l, 1, r))
            else:
                dev_triples_stm.append((r, 1, l))
                dev_triples_stm.append((l, 1, r))

        self.train_triples_stm = train_triples_stm
        self.dev_triples_stm = dev_triples_stm

        attract_constraints = list(self.attract_constraints)
        repel_constraints = list(self.repel_constraints)

        lim_att = int(len(attract_constraints) * .99)
        lim_rep = int(len(repel_constraints) * .99)

        train_triples = defaultdict(list)
        all_triples = defaultdict(list)

        for i, (l, r) in enumerate(attract_constraints):
            if i < lim_att:
                train_triples[(r, 0)].append(l)
                train_triples[(l, 0)].append(r)
            all_triples[(r, 0)].append(l)
            all_triples[(l, 0)].append(r)

        for i, (l, r) in enumerate(repel_constraints):
            if i < lim_rep:
                train_triples[(r, 1)].append(l)
                train_triples[(l, 1)].append(r)
            all_triples[(r, 1)].append(l)
            all_triples[(l, 1)].append(r)

        train_triples = [k + (v,) for k, v in train_triples.items()]
        dev_triples_syn, dev_triples_ant = [], []
        for i, (l, r) in enumerate(attract_constraints[lim_att:]):
            dev_triples_syn.append((l, r, 0, all_triples[(l, 0)], all_triples[(r, 0)]))
        for i, (l, r) in enumerate(repel_constraints[lim_rep:]):
            dev_triples_ant.append((l, r, 1, all_triples[(l, 1)], all_triples[(r, 1)]))

        self.train_triples = train_triples
        self.dev_triples_syn = dev_triples_syn
        self.dev_triples_ant = dev_triples_ant

    def stm_prep(self, pe, ne):

        # positive
        e1, rel, e2 = zip(*pe)
        rel = [1. - self.label_smoothing_epsilon] * len(rel)
        # random
        e1 = list(e1) * 2
        rel = rel + ([0. + self.label_smoothing_epsilon] * len(rel))
        e2 = list(e2) + [random_different_from(self.vocab_size_src, e) for e in e2]
        # negative
        n1, nrel, n2 = zip(*ne)
        e1 = e1 + list(n1)
        rel = rel + [0. + self.label_smoothing_epsilon] * len(nrel)
        e2 = e2 + list(n2)

        e1 = Variable(LongTensorWrapper(e1))
        rel = Variable(FloatTensorWrapper(rel))
        e2 = Variable(LongTensorWrapper(e2))
        return e1, rel, e2

    def extract_candidates(self):
        new_syns, new_ants = [], []
        new_syns_hard_thr, new_ants_hard_thr = [], []
        for rel, ctr in enumerate([self.attract_ctr_tgt, self.repel_ctr_tgt]):
            ctr_batches = len(ctr) // self.params.ctr_batch_size
            relv = Variable(LongTensorWrapper([rel] * self.params.ctr_batch_size))
            for batch_index in range(0, ctr_batches):

                # do one batch:
                examples = [ctr[x] for x in range(batch_index * self.params.ctr_batch_size, (batch_index + 1) * self.params.ctr_batch_size)]
                e1, e2 = zip(*examples)
                e1v = Variable(LongTensorWrapper(e1))
                e2v = Variable(LongTensorWrapper(e2))
                pred1 = self.discriminator.forward(e1v, relv, e2v).sigmoid()
                pred2 = self.discriminator.forward(e2v, relv, e1v).sigmoid()
                soft_thr = self.confidence_level_syn if not rel else self.confidence_level_ant
                is_above_soft = (pred1 > soft_thr) * (pred2 > soft_thr)
                is_above_hard = (pred1 > 0.5) * (pred2 > 0.5)
                zipall = zip(e1, e2, is_above_soft, is_above_hard)
                new_soft = [(a, b) for a, b, c, d in zipall if c]
                new_hard = [(a, b) for a, b, c, d in zipall if d]
                if not rel:
                    new_syns.extend(new_soft)
                    new_syns_hard_thr.extend(new_hard)
                elif rel:
                    new_ants.extend(new_soft)
                    new_ants_hard_thr.extend(new_hard)

        print("Extracted new", len(new_syns), "synonym pairs and ", len(new_ants), "antonym pairs with a soft treshold.")
        print("Found", len(new_syns_hard_thr), "synonym pairs and ", len(new_ants_hard_thr), "antonym pairs with a hard threshold.")
        return new_syns, new_ants

    def stm_train(self, relnum, stm_batch_size=16, stm_max_iter=10, stm_opt="adam,lr=0.0001", l2_factor=0.):
        stm_train_triples = [t for t in self.train_triples_stm if t[1] == relnum]
        stm_negative_triples = [t for t in self.train_triples_stm if t[1] != relnum]
        stm_batches = int(len(stm_train_triples) / stm_batch_size)

        print("Constraint triples: train", len(stm_train_triples))
        print("Running the optimisation procedure for", stm_max_iter, "iterations...")

        last_time = time.time()
        # set optimizer
        optim_fn, optim_params = get_optimizer(stm_opt + ",weight_decay="+str(l2_factor))
        optimizer = optim_fn(self.STM.parameters(), **optim_params)
        ctr_loss = torch.nn.BCEWithLogitsLoss()
        total_loss, f1 = 0., 0.
        for current_iteration in range(stm_max_iter):
            self.STM.train()
            if current_iteration == 0:
                print("\nStarting epoch:", current_iteration+1)
            else:
                print("Starting epoch:", current_iteration+1, "Last epoch took:", round(time.time() - last_time, 1),
                      "seconds. Train loss:", total_loss, "Eval F1", f1)
                last_time = time.time()
                total_loss = 0.

            random.shuffle(stm_train_triples)
            for batch_index in range(0, stm_batches):

                # do one batch:
                examples = [stm_train_triples[x] for x in range(batch_index * stm_batch_size, (batch_index+1) * stm_batch_size)]
                neg_examples = random.sample(stm_negative_triples, len(examples))
                e1, rel, e2 = self.stm_prep(examples, neg_examples)
                scores = self.STM(e1, e2)
                loss = ctr_loss(scores.squeeze(), rel)
                # apply gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss = total_loss + torch.sum(loss).item()

            total_loss /= stm_batches
            self.STM.eval()
            with torch.no_grad():
                f1 = self.stm_eval(relnum)

        model_path = self.params.out_dir + "stm_model.pth"
        print('saving to {0}'.format(model_path))
        torch.save(self.STM.state_dict(), model_path)

    def stm_eval(self, relnum, stm_batch_size=50):
        dev_triples = self.dev_triples_stm
        dev_count = len(dev_triples)
        stm_batches = int(dev_count / stm_batch_size)
        p, r = [], []

        for batch_index in range(0, stm_batches):
            # do one batch:
            examples = [dev_triples[x] for x in range(batch_index * stm_batch_size, (batch_index + 1) * stm_batch_size)]
            e1, rel, e2 = zip(*examples)
            rel = [1 if rn == relnum else 0 for rn in rel]
            e1 = Variable(LongTensorWrapper(e1))
            rel = Variable(FloatTensorWrapper(rel))
            e2 = Variable(LongTensorWrapper(e2))

            pred = self.STM(e1, e2).sigmoid()
            pred = pred > 0.5
            pred = list(pred.data.cpu().numpy())
            rel = list(rel.long().data.cpu().numpy())
            p.extend(pred)
            r.extend(rel)
        f1_total = f1_score(r, p)
        return f1_total

    def refine_candidates(self, synsants, stm_batch_size=50):

        new_synsants = []
        batches = len(synsants) // stm_batch_size
        for batch_index in range(0, batches):
            # do one batch:
            examples = [synsants[x] for x in range(batch_index * stm_batch_size, (batch_index + 1) * stm_batch_size)]
            e1, e2 = zip(*examples)
            e1v = Variable(LongTensorWrapper(e1))
            e2v = Variable(LongTensorWrapper(e2))
            pred1 = self.STM(e1v, e2v, tgt=True).sigmoid()
            pred2 = self.STM(e2v, e1v, tgt=True).sigmoid()
            pred = (pred1.squeeze() > 0.5) * (pred2.squeeze() > 0.5)
            new_synsants.extend([ex for ex, bv in zip(examples, pred) if bv])

        print("Filtered in new", len(new_synsants), "pairs.")

        return new_synsants

    def export(self, syn, ant, afx=""):
        fout = codecs.open(self.params.out_dir + "new_" + afx, "w", encoding="utf8")
        for l, r in syn:
            fout.write(l + " " + r + " syn\n")
        for l, r in ant:
            fout.write(l + " " + r + " ant\n")
        fout.close()
