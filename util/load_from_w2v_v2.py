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

    model = Word2Vec.load_word2vec_format('../embeddings/baike_vector.bin', binary=True)

    vocab_celebrity = load(serialization_data_path, 'celebrity_vocabulary_v2.pkl')
    vocab_movie = load(serialization_data_path, 'movie_vocabulary_v2.pkl')
    vocab_restaurant = load(serialization_data_path, 'restaurant_vocabulary_v2.pkl')
    vocab_tvShow = load(serialization_data_path, 'tvShow_vocabulary_v2.pkl')
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    emb_c = np.random.rand(len(vocab_celebrity)+1, 300)
    emb_m = np.random.rand(len(vocab_movie)+1, 300)
    emb_r = np.random.rand(len(vocab_restaurant)+1, 300)
    emb_t = np.random.rand(len(vocab_tvShow)+1, 300)

    # swap the word2vec weights with the embedded weights
    for i, w in vocab_celebrity.items():
        if w not in d:
            continue
        emb_c[i, :] = weights[d[w], :]

    for i, w in vocab_movie.items():
        if w not in d:
            continue
        emb_m[i, :] = weights[d[w], :]

    for i, w in vocab_restaurant.items():
        if w not in d:
            continue
        emb_r[i, :] = weights[d[w], :]

    for i, w in vocab_tvShow.items():
        if w not in d:
            continue
        emb_t[i, :] = weights[d[w], :]

    np.save(open('../embeddings/celebrity_300_dim_v2.embeddings', 'wb'), emb_c)
    np.save(open('../embeddings/movie_300_dim_v2.embeddings', 'wb'), emb_m)
    np.save(open('../embeddings/restaurant_300_dim_v2.embeddings', 'wb'), emb_r)
    np.save(open('../embeddings/tvShow_300_dim_v2.embeddings', 'wb'), emb_t)
