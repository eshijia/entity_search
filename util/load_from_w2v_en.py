# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import random
import numpy as np

import cPickle

from gensim.models import Word2Vec

random.seed(42)

serialization_data_path = '../serialization_data'
embedding_path = '../embeddings'


def load(path, name):
    return cPickle.load(open(os.path.join(path, name), 'rb'))


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]

if __name__ == '__main__':

    model = Word2Vec.load_word2vec_format('../embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    vocab_SemSearch_ES = load(serialization_data_path, 'SemSearch_ES_vocabulary.pkl')
    vocab_ListSearch = load(serialization_data_path, 'ListSearch_vocabulary.pkl')
    vocab_INEX_LD = load(serialization_data_path, 'INEX_LD_vocabulary.pkl')
    vocab_QALD2 = load(serialization_data_path, 'QALD2_vocabulary.pkl')
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    emb_s = np.random.rand(len(vocab_SemSearch_ES)+1, 300)
    emb_l = np.random.rand(len(vocab_ListSearch)+1, 300)
    emb_i = np.random.rand(len(vocab_INEX_LD)+1, 300)
    emb_q = np.random.rand(len(vocab_QALD2)+1, 300)

    # swap the word2vec weights with the embedded weights
    for i, w in vocab_SemSearch_ES.items():
        if w not in d:
            continue
        emb_s[i, :] = weights[d[w], :]

    for i, w in vocab_ListSearch.items():
        if w not in d:
            continue
        emb_l[i, :] = weights[d[w], :]

    for i, w in vocab_INEX_LD.items():
        if w not in d:
            continue
        emb_i[i, :] = weights[d[w], :]

    for i, w in vocab_QALD2.items():
        if w not in d:
            continue
        emb_q[i, :] = weights[d[w], :]

    np.save(open('../embeddings/SemSearch_ES_300_dim.embeddings', 'wb'), emb_s)
    np.save(open('../embeddings/ListSearch_300_dim.embeddings', 'wb'), emb_l)
    np.save(open('../embeddings/INEX_LD_300_dim.embeddings', 'wb'), emb_i)
    np.save(open('../embeddings/QALD2_300_dim.embeddings', 'wb'), emb_q)
