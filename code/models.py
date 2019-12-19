#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from torch import nn
from dictionary import Dictionary
from itertools import chain


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class Generator(nn.Module):

    def __init__(self, params):
        super(Generator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.gen_layers
        self.dis_hid_dim = params.gen_hid_dim
        self.dis_dropout = params.gen_dropout
        self.dis_input_dropout = params.gen_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = self.emb_dim if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x)


def normalize_embeddings(emb, types):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            emb.sub_(emb.mean(1, keepdim=True).expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)


def build_model(params, ar_module):
    """
    Build all components of the model.
    """

    constraint_indices = list(set(list(chain(*ar_module.synonyms)) + list(chain(*ar_module.antonyms))))
    constraint_words = [ar_module.inverted_index[i] for i in constraint_indices]
    dico = Dictionary(dict(zip(constraint_indices, constraint_words)), dict(zip(constraint_words, constraint_indices)))

    for emb in [ar_module.model.init_W, ar_module.model.dynamic_W]:
        emb.weight.requires_grad = False
        normalize_embeddings(emb.weight.data, params.normalize_embeddings)

    # mapping
    mapping = Generator(params)
    # discriminator
    discriminator = Discriminator(params) if params.adversarial else None

    # cuda
    if params.cuda:
        mapping.cuda()
        if params.adversarial:
            discriminator.cuda()
    return constraint_indices, dico, mapping, discriminator
