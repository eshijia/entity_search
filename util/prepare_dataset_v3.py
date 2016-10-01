# -*- coding: utf-8 -*-
from optparse import OptionParser

import codecs
import jieba
import os
import re
import cPickle

data_path = '../data'
serialization_data_path = '../serialization_data'
jieba.load_userdict('../data/dictionary.txt')

if __name__ == '__main__':

    parser = OptionParser()
    vocab = set()
    vocabulary = dict()
    answers = dict()
    ans2idx = dict()
    train = list()
    test = list()
    parser.add_option("-t", "--type", dest="type_filename", type="string",
                      help="The type(celebrity/movie/restaurant/tvShow) of the file name")
    (options, args) = parser.parse_args()

    file_class = options.type_filename
    train_set = codecs.open(os.path.join(data_path, file_class + '.TRAINSET.txt'), 'rb', 'gb18030')
    entity_set = codecs.open(os.path.join(data_path, file_class + '.ENTITYSET.txt'), 'rb', 'gb18030')
    test_set = codecs.open(os.path.join(data_path, file_class + '.GROUNDTRUTH.txt'), 'rb', 'gb18030')

    # for line in train_set.readlines():
    #     line = line.strip().split('\t')
    #     query_words = jieba.cut(line[0])
    #     for query_word in query_words:
    #         vocab.add(query_word)
    #     for term in line[1:]:
    #         words = jieba.cut(re.sub(r'[\(\)]', '', ''.join(term.split(':')[:-1])))
    #         for word in words:
    #             vocab.add(word)
    for line in train_set.readlines() + test_set.readlines() + entity_set.readlines():
        line = line.strip().split('\t')
        for term in line:
            words = jieba.cut(term)
            for word in words:
                vocab.add(word)
    for i, elem in enumerate(vocab):
        vocabulary[i+1] = elem
    print "Vocabulary: %d" % len(vocabulary)

    entity_set.seek(0)
    test_set.seek(0)
    train_set.seek(0)

    vocab2idx = dict((value, key) for key, value in vocabulary.iteritems())

    for i, line in enumerate(entity_set.readlines()):
        temp_dict = {}
        line = line.strip()
        ans2idx[line] = i
        real_entity = re.sub(r'\(.+?\)', '', line)
        entity_des = re.findall(r'\((.+?)\)', line)
        entity_description = entity_des[0] if entity_des else "*"
        temp_dict['entity'] = [vocab2idx[word] for word in jieba.cut(real_entity)]
        temp_dict['des'] = [vocab2idx[word] for word in jieba.cut(entity_description)]
        answers[i] = temp_dict
    print "Answers: %d" % len(answers)
    print "Answer IDs: %d" % len(ans2idx)

    for line in train_set.readlines():
        temp_dict = dict()
        line = line.strip().split('\t')
        temp_good_answers = list()
        temp_bad_answers = list()
        temp_dict['question'] = [vocab2idx[word] for word in jieba.cut(line[0])]
        for candidate in line[1:]:
            candidate = candidate.split(':')
            if candidate[-1] == '1':
                temp_good_answers.append(':'.join(candidate[:-1]))
            else:
                temp_bad_answers.append(':'.join(candidate[:-1]))
        temp_dict['good_answers'] = [ans2idx[ans] for ans in temp_good_answers]
        temp_dict['bad_answers'] = [ans2idx[ans] for ans in temp_bad_answers]
        train.append(temp_dict)
    print "Train samples: %d" % len(train)

    for line in test_set.readlines():
        temp_dict = dict()
        line = line.strip().split('\t')
        temp_good_answers = list()
        temp_bad_answers = list()
        temp_dict['question'] = [vocab2idx[word] for word in jieba.cut(line[0])]
        for candidate in line[1:]:
            candidate = candidate.split(':')
            if candidate[-1] == '1':
                temp_good_answers.append(':'.join(candidate[:-1]))
            else:
                temp_bad_answers.append(':'.join(candidate[:-1]))
        temp_dict['good_answers'] = [ans2idx[ans] for ans in temp_good_answers]
        temp_dict['bad_answers'] = [ans2idx[ans] for ans in temp_bad_answers]
        test.append(temp_dict)
    print "Test samples: %d" % len(test)

    with open(os.path.join(serialization_data_path, file_class + '_vocabulary_v3.pkl'), 'wb') as output_vocab:
        cPickle.dump(vocabulary, output_vocab, -1)

    with open(os.path.join(serialization_data_path, file_class + '_answers_v3.pkl'), 'wb') as output_answers:
        cPickle.dump(answers, output_answers, -1)

    with open(os.path.join(serialization_data_path, file_class + '_train_v3.pkl'), 'wb') as output_train:
        cPickle.dump(train, output_train, -1)

    with open(os.path.join(serialization_data_path, file_class + '_ans2idx_v3.pkl'), 'wb') as output_ans2idx:
        cPickle.dump(ans2idx, output_ans2idx, -1)

    with open(os.path.join(serialization_data_path, file_class + '_test_v3.pkl'), 'wb') as output_test:
        cPickle.dump(test, output_test, -1)
