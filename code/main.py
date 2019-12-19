#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
import argparse
import torch
import pickle
import logging
import random

from models import build_model
from trainer_batch import Trainer
from evaluator import Evaluator
from attract_repel import ExperimentRun
from constraint_predictor import CtrPredictor

parser = argparse.ArgumentParser(description='Cross-lingual zero-shot specialization via lexical relation induction')

# META
parser.add_argument("--seed", type=int, default=3, help="Initialization seed")
parser.add_argument("--verbose", type=str, default="debug", help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
parser.add_argument("--bootstrap_iter", type=int, default=5, help="Maximum number of iterations of bootstrapping")
parser.add_argument('--restore_mapping', action='store_true', help='restore pre-trained mapping function')

# ATTRACT-REPEL
parser.add_argument("--distributional_vectors", type=str, required=True, help="Distributional vectors file")
parser.add_argument("--target_vectors", type=str, required=True, help="File of aligned vectors from another language")
parser.add_argument("--target_lang", type=str, required=True, help="Name of target language")
parser.add_argument("--attract_constraints", nargs='+', required=True, help="Attract constraints")
parser.add_argument("--repel_constraints", nargs='+', required=True, help="Repel constraints")
parser.add_argument("--attract_tgtctr", nargs='+', required=True, help="Target attract constraints")
parser.add_argument("--repel_tgtctr", nargs='+', required=True, help="Target repel constraints")
parser.add_argument("--out_dir", type=str, required=True, help="Directory where to store experiment results")

parser.add_argument('--no_simlex', action='store_true', help='print SimLex')
parser.add_argument("--attract_margin", type=float, default=0.6, help="Attract margin")
parser.add_argument("--repel_margin", type=float, default=0.0, help="Repel margin")
parser.add_argument("--batch_size_ar", type=int, default=50, help="Batch size")
parser.add_argument("--max_iter", type=int, default=5, help="Maximum number of iterations")
parser.add_argument("--l2_reg_constant", type=float, default=1e-9, help="L2 regularization factor")

# POST-SPECIALIZATION
parser.add_argument("--adversarial", action="store_true", help="Train mapping adversarially")
# embs
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="renorm", help="Normalize embeddings before training")
# mapping
parser.add_argument("--gen_layers", type=int, default=3, help="Generator layers")
parser.add_argument("--gen_hid_dim", type=int, default=2048, help="Generator hidden layer dimensions")
parser.add_argument("--gen_dropout", type=float, default=0.2, help="Generator dropout")
parser.add_argument("--gen_input_dropout", type=float, default=0.2, help="Generator input dropout")
parser.add_argument("--gen_lambda", type=float, default=1, help="Generator loss feedback coefficient")
parser.add_argument("--sim_loss", type=str, default="max_margin", help="Similarity loss: mse or max_margin")
parser.add_argument("--sim_margin", type=float, default=1, help="Similarity margin (for max_margin losse)")
parser.add_argument("--sim_neg", type=int, default=25, help="Similarity negative examples (for max_margin loss)")
parser.add_argument("--sim_lambda", type=float, default=1, help="Similarity loss feedback coefficient")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.5, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Epoch_size")
parser.add_argument("--patience", type=int, default=3, help="Patience")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--sim_optimizer", type=str, default="sgd,lr=0.1", help="Similarity optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

# CONSTRAINT PREDICTION
parser.add_argument("--ctr_dis", type=str, default="ConvE", help="Discriminator architecture for constraint prediction")
parser.add_argument("--ctr_opt", type=str, default="adam,lr=0.0001", help="Discriminator learning rate")
parser.add_argument("--ctr_batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--ctr_hid", type=int, default=600, help="Hidden size")
parser.add_argument("--ctr_max_iter", type=int, default=100, help="Maximum number of iterations")
parser.add_argument("--ctr_l2", type=float, default=0., help="Discriminator learning rate")

params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert os.path.isfile(params.distributional_vectors)
if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    pickle.dump(params, open(os.path.join(params.out_dir, 'params.pkl'), 'wb'))

    # create logger
    logging.basicConfig(level=getattr(logging, params.verbose.upper()))
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logging.info('The experiment will be stored in %s' % params.out_dir)


initialize_exp(params)

logging.info('----> ATTRACT-REPEL <----\n\n')
attract_repel_src = ExperimentRun(params, is_target=False)
attract_repel_tgt = ExperimentRun(params, is_target=True)
if not params.restore_mapping:
    attract_repel_src.attract_repel()

logging.info('----> POST-SPECIALIZATION <----\n\n')
ps_key = "xling_postspec_"

ctr_idx, dico, mapping, discriminator = build_model(params, attract_repel_src)
trainer = Trainer(ctr_idx, dico, attract_repel_src, mapping, discriminator, params)
evaluator = Evaluator(trainer, attract_repel_src.inverted_index, attract_repel_src.language)
if not params.restore_mapping:
    trainer.run_experiment(params, evaluator, afx=ps_key)
trainer.reload_best(afx=ps_key)
trainer.xling(params, attract_repel_tgt, afx=ps_key + params.target_lang)

logging.info('----> NEW CONSTRAINTS PREDICTION <----\n\n')
ps_key = "constraints_" + params.target_lang
ctr_predictor = CtrPredictor(trainer, attract_repel_src,
                             attract_repel_tgt, params.ctr_dis)
new_synonyms, new_antonyms = ctr_predictor.routine()
ctr_predictor.export(new_synonyms, new_antonyms, afx=ps_key)

attract_repel_tgt.add_new_constraints(new_synonyms, new_antonyms)
attract_repel_tgt.attract_repel()
attract_repel_tgt.print_word_vectors("_" + params.target_lang)

params.epoch_size = 200000
params.n_epochs = 2
ps_key = "stm_postspec_" + params.target_lang
tgt_ctr_idx, tgt_dico, tgt_mapping, tgt_discriminator = build_model(params, attract_repel_tgt)
tgt_trainer = Trainer(tgt_ctr_idx, tgt_dico, attract_repel_tgt, tgt_mapping, tgt_discriminator, params)
tgt_evaluator = Evaluator(tgt_trainer, attract_repel_tgt.inverted_index, attract_repel_tgt.language)
tgt_trainer.run_experiment(params, tgt_evaluator, afx=ps_key)
tgt_trainer.reload_best(afx=ps_key)
tgt_trainer.xling(params, attract_repel_tgt, afx=ps_key)
