# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import random
import jieba
from time import strftime, gmtime

import pickle
import codecs
from keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
from heapq import nlargest
from gensim.models import Word2Vec

from keras_models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)


class Evaluator:
    def __init__(self, conf=None):
        serialization_data_path = 'serialization_data'
        self.path = serialization_data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.type = conf.get('type', None)
        self.answers = self.load(self.type + '_answers_v2.pkl')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load(self.type + '_vocabulary_v2.pkl')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        model.save_weights('models/' + self.type + '_weights_epoch_%d_v2.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/' + self.type + '_weights_epoch_%d_v2.h5' % epoch), 'Weights at epoch %d not found' % epoch
        model.load_weights('models/' + self.type + '_weights_epoch_%d_v2.h5' % epoch)

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def print_time(self):
        print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

    def train(self, model):
        loss_record = list()
        eval_every = self.params.get('eval_every', None)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)
        split = self.params.get('validation_split', 0)

        training_set = self.load(self.type + '_train_v2.pkl')

        questions = list()
        good_answers = list()
        bad_answer_candidates = list()

        for q in training_set:
            questions += [q['question']] * len(q['good_answers'])
            good_answers += [self.answers[i] for i in q['good_answers']]
            bad_answer_candidates += [q['bad_answers']] * len(q['good_answers'])

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)
        print("Questions: ", len(questions))
        print("Good answers: ", len(good_answers))

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch+1):
            # bad_answers = np.roll(good_answers, random.randint(10, len(questions) - 10))
            # bad_answers = good_answers.copy()
            # random.shuffle(bad_answers)
            # bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))
            bad_answers = self.pada([self.answers[random.choice(candidate)] for candidate in bad_answer_candidates])
            # print("Bad answers: ", len(bad_answers))

            # shuffle questions
            # zipped = zip(questions, good_answers)
            # random.shuffle(zipped)
            # questions[:], good_answers[:] = zip(*zipped)

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            hist = model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size, validation_split=split)
            loss_record.append(hist.history['loss'][0])
            # if hist.history['val_loss'][0] < val_loss['loss']:
            #     val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if eval_every is not None and i % eval_every == 0:
                self.get_map(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)
        return loss_record

    ##### Evaluation #####
    def get_map(self, model, evaluate_all=False):
        top1s = list()
        maps = list()

        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            ap, h1 = 0, 0

            for i, d in enumerate(data):
                if evaluate_all:
                    self.prog_bar(i, len(data))

                indices = d['good_answers'] + d['bad_answers']
                answers = self.pada([self.answers[index] for index in indices])
                question = self.padq([d['question']] * len(indices))

                n_good = len(d['good_answers'])
                sims = model.predict([question, answers], batch_size=len(indices)).flatten()
                r = rankdata(sims, method='ordinal')

                target_rank = np.asarray(r[:n_good])
                num_candidate = len(sims)
                ground_rank = num_candidate - target_rank + 1
                ground_rank.sort()
                one_ap = 0
                for rank in xrange(n_good):
                    one_ap += (rank + 1) / float(ground_rank[rank])
                one_ap /= n_good

                ap += one_ap
                h1 += 1 if np.argmax(sims) < n_good else 0

                # max_r = np.argmax(sims)
                # max_n = np.argmax(sims[:n_good])

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                # c_1 += 1 if max_r == max_n else 0
                # c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = h1 / float(len(data))
            mean_ap = ap / float(len(data))

            del data

            if evaluate_all:
                print('Top-1 Precision: %f' % top1)
                print('MAP: %f' % mean_ap)

            top1s.append(top1)
            maps.append(mean_ap)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MAP: {}'.format(maps))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            map_threshold = evaluate_all_threshold.get('map', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= map_threshold for x in maps])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= map_threshold for x in maps])

            if evaluate_all:
                return self.get_map(model, evaluate_all=True)

        return top1s, maps

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in [self.type + '_test_v2.pkl']])
        return self._eval_sets

    def max_sim(self, left_list, right_list, w2v_model):
        max_cos = -1
        for i in range(len(left_list)):
            for j in range(len(right_list)):
                try:
                    local_sim = w2v_model.similarity(left_list[i], right_list[j])
                    max_cos = local_sim if local_sim > max_cos else max_cos
                except KeyError:
                    pass
        return max_cos

    def make_submit(self, model, dev, submit_file):
        lines = dev.readlines()
        target_lines = list()
        data = self.eval_sets().values()[0]
        for i, d in enumerate(data):
            terms = lines[i].strip('\n').split('\t')
            query = terms[0]
            entities = terms[1:]
            num_candidate = len(entities)
            index_entities = xrange(num_candidate)

            indices = d['answers']
            answers = self.pada([self.answers[i] for i in indices])
            question = self.padq([d['question']] * len(indices))

            sims = model.predict([question, answers], batch_size=100).flatten()
            print(len(sims))
            r = rankdata(sims, method='ordinal')
            index_candidates = nlargest(num_candidate, index_entities, key=lambda j: r[j])
            for index_candidate in index_candidates:
                query = query + '\t' + entities[index_candidate]
            query += '\n'
            target_lines.append(query)
        submit_file.writelines(target_lines)

        del data

    def make_submit_v2(self, model, dev, submit_file, w2v_model):
        lines = dev.readlines()
        target_lines = list()
        data = self.eval_sets().values()[0]
        for i, d in enumerate(data):
            terms = lines[i].strip('\n').split('\t')
            query = terms[0]
            query_word_list = [x for x in jieba.cut(query)]
            entities = terms[1:]
            num_candidate = len(entities)
            sim_candidate = list()

            for entity in entities:
                sim_candidate.append(self.max_sim(query_word_list, [y for y in jieba.cut(entity)], w2v_model))

            sim_candidate = np.asarray(sim_candidate)

            index_entities = xrange(num_candidate)

            indices = d['answers']
            answers = self.pada([self.answers[i] for i in indices])
            question = self.padq([d['question']] * len(indices))

            sims = model.predict([question, answers], batch_size=100).flatten()
            sims = sim_candidate * 0.6 + sims * 0.4
            print(len(sims))
            r = rankdata(sims, method='ordinal')
            index_candidates = nlargest(num_candidate, index_entities, key=lambda j: r[j])
            for index_candidate in index_candidates:
                query = query + '\t' + entities[index_candidate]
            query += '\n'
            target_lines.append(query)
        submit_file.writelines(target_lines)

        del data


if __name__ == '__main__':
    conf = {
        'type': 'movie',
        'question_len': 8,
        'answer_len': 1,
        'n_words': 25458,  # len(vocabulary) + 1
        'margin': 0.02,

        'training_params': {
            'save_every': 1000,
            # 'eval_every': 10,
            'batch_size': 32,
            'nb_epoch': 100,
            'validation_split': 0,
            'optimizer': 'adam',
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.5,
            },
        },

        'model_params': {
            'n_embed_dims': 300,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000, # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141, # * 2

            # 'initial_embed_weights': np.load('embeddings/movie_300_dim_v2.embeddings'),
        },

        'similarity_params': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(conf)

    ##### Define model ######
    model = EmbeddingModel(conf)
    model.compile(optimizer='Adam')

    model2 = EmbeddingModel(conf)
    model2.compile(optimizer='Adagrad')

    model3 = EmbeddingModel(conf)
    model3.compile(optimizer='SGD')

    model4 = EmbeddingModel(conf)
    model4.compile(optimizer='RMSprop')

    model5 = EmbeddingModel(conf)
    model5.compile(optimizer='Adadelta')

    import numpy as np

    # train the model
    target = evaluator.train(model)
    target2 = evaluator.train(model2)
    target3 = evaluator.train(model3)
    target4 = evaluator.train(model4)
    target5 = evaluator.train(model5)

    plt.figure(1)

    axes = plt.gca()
    x_min = 1
    x_max = len(target)
    axes.set_xlim([x_min, x_max])
    axes.set_xticks(np.linspace(x_min, x_max, 10))

    plt.scatter(np.arange(1, len(target) + 1), np.asarray(target), color='g', s=10)
    plt.scatter(np.arange(1, len(target2) + 1), np.asarray(target2), color='b', s=10)
    plt.scatter(np.arange(1, len(target3) + 1), np.asarray(target3), color='r', s=10)
    plt.scatter(np.arange(1, len(target4) + 1), np.asarray(target4), color='y', s=10)
    plt.scatter(np.arange(1, len(target5) + 1), np.asarray(target5), color='k', s=10)
    plt.plot(np.arange(1, len(target) + 1), np.asarray(target), color='g', label='Adam')
    plt.plot(np.arange(1, len(target2) + 1), np.asarray(target2), color='b', label='Adagrad')
    plt.plot(np.arange(1, len(target3) + 1), np.asarray(target3), color='r', label='SGD')
    plt.plot(np.arange(1, len(target4) + 1), np.asarray(target4), color='y', label='RMSprop')
    plt.plot(np.arange(1, len(target5) + 1), np.asarray(target5), color='k', label='Adadelta')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend(loc=0, fontsize='x-small')
    # plt.show()
    plt.savefig('figure4.png')
