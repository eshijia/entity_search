# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import Word2Vec
from heapq import nlargest

from optparse import OptionParser

import numpy as np
import codecs
import jieba

# line = devset.readline().strip('\n').split('\t')
# query = line[0]
# entity_list = line[1:]
# print entity_list[37]
# left_list = [x for x in jieba.cut(query)]
#
# right_list1 = [y for y in jieba.cut(entity_list[0])]
# right_list2 = jieba.cut(entity_list[38])
# model = Word2Vec.load_word2vec_format('celebrity-v3.bin', binary=True)
#
# local_sim = model.similarity(left_list[2], right_list1[0])
# print local_sim


def max_sim(left_list, right_list, w2v_model):
    max_cos = -1
    for i in range(len(left_list)):
        for j in range(len(right_list)):
            try:
                local_sim = w2v_model.similarity(left_list[i], right_list[j])
                max_cos = local_sim if local_sim > max_cos else max_cos
            except KeyError:
                pass
    return max_cos


def avg_sim(left_list, right_list, w2v_model):
    sim_list = [-1.0]
    for i in range(len(left_list)):
        for j in range(len(right_list)):
            try:
                local_sim = w2v_model.similarity(left_list[i], right_list[j])
                sim_list.append(local_sim)
            except KeyError:
                pass
    return np.mean(sim_list)


def make_submit(w2v_model, test_file, submit_file):
    lines = test_file.readlines()
    target_lines = list()
    for line in lines:
        terms = line.strip('\n').split('\t')
        query = terms[0]
        query_word_list = [x for x in jieba.cut(query)]
        entities = terms[1:]
        num_candidate = len(entities)
        sim_candidate = list()
        for entity in entities:
            sim_candidate.append(max_sim(query_word_list, [y for y in jieba.cut(entity)], w2v_model))
        index_entities = xrange(num_candidate)
        index_candidates = nlargest(num_candidate, index_entities, key=lambda i: sim_candidate[i])
        for index_candidate in index_candidates:
            query = query + '\t' + entities[index_candidate]
        query += '\n'
        target_lines.append(query)
    submit_file.writelines(target_lines)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-v", "--vector", dest="vector_filename", type="string",
                      help="Your word vector file name")
    parser.add_option("-o", "--output", dest="submit_filename", type="string",
                      help="The output file to be submitted")
    parser.add_option("-t", "--test", dest="test_filename", type="string",
                      help="The test file to predict")
    (options, args) = parser.parse_args()

    dev_set = codecs.open(options.test_filename, 'rb', 'gb18030')
    submit = codecs.open(options.submit_filename, 'wb', 'gb18030')
    model = Word2Vec.load_word2vec_format(options.vector_filename, binary=True)
    make_submit(model, dev_set, submit)
