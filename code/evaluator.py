# -*- coding: utf-8 -*-

from logging import getLogger
import torch
from attract_repel import simlex_analysis
logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, ar_ii, ar_lang):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.dico = trainer.dico
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.train_idx = trainer.train_idx
        self.ar_dico = ar_ii
        self.language = ar_lang

    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        ids = torch.LongTensor(self.train_idx)
        ids = ids.cuda() if self.params.cuda else ids
        self.mapping.eval()
        with torch.no_grad():
            src_emb = self.mapping(self.src_emb(ids)).data
        tgt_emb = self.tgt_emb(ids).data

        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        mean_cosine = (src_emb * tgt_emb).sum(1).mean()

        mean_cosine = mean_cosine.item()
        logger.info("Mean cosine: %.5f" % mean_cosine)
        to_log['mean_cosine'] = mean_cosine

    def simlex_scores(self, to_log, language):
        # get normalized embeddings
        self.mapping.eval()
        with torch.no_grad():
            src_emb = self.mapping(self.src_emb.weight).data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        keyed_src_emb = {self.ar_dico[i]: v for i, v in enumerate(src_emb)}

        simlex_score, simlex_coverage = simlex_analysis(keyed_src_emb, language)
        logger.info("SimLex score for %s is %s coverage: %s / 999", language, str(simlex_score), str(simlex_coverage))
        to_log['simlex_score'] = simlex_score

        if language == "english":
            simverb_score, simverb_coverage = simlex_analysis(keyed_src_emb, language, source="simverb")
            logger.info("SimVerb score for english is: %s coverage %s / 3500", str(simverb_score), str(simverb_coverage))
            to_log['simverb_score'] = simverb_score

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.dist_mean_cosine(to_log)
        if not self.params.no_simlex:
            self.simlex_scores(to_log, self.language)
