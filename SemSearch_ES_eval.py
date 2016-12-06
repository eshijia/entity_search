from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime

import pickle

from keras.optimizers import Adam, SGD
from scipy.stats import rankdata

from keras_models_en import *

random.seed(42)


class Evaluator:
    def __init__(self, conf=None):
        serialization_data_path = 'serialization_data'
        self.path = serialization_data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.answers = self.load('SemSearch_ES_answers.pkl')
        self._vocab = self.load('SemSearch_ES_vocabulary.pkl')
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('SemSearch_ES_vocabulary.pkl')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/SemSearch_ES/'):
            os.makedirs('models/SemSearch_ES/')
        model.save_weights('models/SemSearch_ES/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/SemSearch_ES/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
        model.load_weights('models/SemSearch_ES/weights_epoch_%d.h5' % epoch)

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
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)
        split = self.params.get('validation_split', 0)

        dataset = self.load('SemSearch_ES_train.pkl')
        test_set = self.load('SemSearch_ES_test.pkl')
        query2id = self.load('SemSearch_ES_query2id.pkl')
        raw_sets = [dataset[i: i+5] for i in range(0, len(dataset), 5)]
        training_set = raw_sets[0] + raw_sets[1] + raw_sets[2] + raw_sets[3]
        eval_sets = []
        for sample in raw_sets[4]:
            test_sample = dict()
            test_query = sample['query']
            test_query_id = query2id[' '.join([self._vocab[idx] for idx in test_query])]
            test_sample.update(query=test_query,
                               good_answers=test_set[test_query_id]['good_answers'],
                               bad_answers=test_set[test_query_id]['bad_answers'])
            eval_sets.append(test_sample)
        self._eval_sets = dict([('SemSearch_ES', eval_sets)])

        questions = list()
        good_answers = list()

        for q in training_set:
            questions += [q['query']] * len(q['good_answers'])
            good_answers += [self.answers[i] for i in q['good_answers']]

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)
        print(len(good_answers))

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch+1):
            # sample from all answers to get bad answers
            bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size, validation_split=split)

            # if hist.history['val_loss'][0] < val_loss['loss']:
            #     val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)

        return val_loss

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['SemSearch_ES_train.pkl']])
        return self._eval_sets

    def get_map(self, model, evaluate_all=False):
        top1s = list()
        top10s = list()
        top20s = list()
        maps = list()

        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            ap, h1, h10, h20, p10, p20 = 0, 0, 0, 0, 0, 0

            for i, d in enumerate(data):
                if evaluate_all:
                    self.prog_bar(i, len(data))

                indices = d['good_answers'] + d['bad_answers']
                answers = self.pada([self.answers[index] for index in indices])
                question = self.padq([d['query']] * len(indices))

                n_good = len(d['good_answers'])
                sims = model.predict([question, answers])
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
                h10 += 1 if set(list(ground_rank - 1)).intersection(set(range(10))) else 0
                h20 += 1 if set(list(ground_rank - 1)).intersection(set(range(20))) else 0

                one_p10 = 0
                one_p20 = 0
                for rank in ground_rank:
                    if rank <= 10:
                        one_p10 += 1
                    if rank <= 20:
                        one_p20 += 1
                one_p10 /= 10.0
                one_p20 /= 20.0
                p10 += one_p10
                p20 += one_p20

                # max_r = np.argmax(sims)
                # max_n = np.argmax(sims[:n_good])

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                # c_1 += 1 if max_r == max_n else 0
                # c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = h1 / float(len(data))
            top10 = h10 / float(len(data))
            top20 = h20 / float(len(data))
            mean_ap = ap / float(len(data))
            p_at_10 = p10 / float(len(data))
            p_at_20 = p20 / float(len(data))

            del data

            if evaluate_all:
                print('Top-1 Precision: %f' % top1)
                print('Hit@10 Precision: %f' % top10)
                print('Hit@20 Precision: %f' % top20)
                print('MAP: %f' % mean_ap)
                print('P@10: %f' % p_at_10)
                print('P@20: %f' % p_at_20)

            top1s.append(top1)
            top10s.append(top10)
            top20s.append(top20)
            maps.append(mean_ap)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('Hit@10 Precision: {}'.format(top10s))
            print('Hit@20 Precision: {}'.format(top20s))
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

        return top1s, top10s, top20s, maps

    def get_mrr(self, model, evaluate_all=False):
        top1s = list()
        mrrs = list()

        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0

            for i, d in enumerate(data):
                if evaluate_all:
                    self.prog_bar(i, len(data))

                indices = d['good'] + d['bad']
                answers = self.pada([self.answers[i] for i in indices])
                question = self.padq([d['question']] * len(indices))

                n_good = len(d['good'])
                sims = model.predict([question, answers])
                r = rankdata(sims, method='max')

                max_r = np.argmax(sims)
                max_n = np.argmax(sims[:n_good])

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                c_1 += 1 if max_r == max_n else 0
                c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = c_1 / float(len(data))
            mrr = c_2 / float(len(data))

            del data

            if evaluate_all:
                print('Top-1 Precision: %f' % top1)
                print('MRR: %f' % mrr)

            top1s.append(top1)
            mrrs.append(mrr)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MRR: {}'.format(mrrs))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            mrr_theshold = evaluate_all_threshold.get('mrr', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

            if evaluate_all:
                return self.get_mrr(model, evaluate_all=True)

        return top1s, mrrs


if __name__ == '__main__':

    conf = {
        'question_len': 10,
        'answer_len': 10,
        'n_words': 12457,  # len(vocabulary) + 1
        'margin': 0.05,

        'training_params': {
            'print_answers': False,
            'save_every': 1,
            'batch_size': 64,
            'nb_epoch': 100,
            'validation_split': 0,
            'optimizer': 'adam',
        },

        'model_params': {
            'n_embed_dims': 300,
            'n_hidden': 200,

            # convolution
            'nb_filters': 500,  # * 4
            'conv_activation': 'tanh',

            # recurrent
            'n_lstm_dims': 141,  # * 2

            'initial_embed_weights': np.load('embeddings/SemSearch_ES_300_dim.embeddings'),
            'similarity_dropout': 0.25,
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
    optimizer = conf.get('training_params', dict()).get('optimizer', 'rmsprop')
    model.compile(optimizer=optimizer)

    # save embedding layer
    # evaluator.load_epoch(model, 7)
    # embedding_layer = model.prediction_model.layers[2].layers[2]
    # weights = embedding_layer.get_weights()[0]
    # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

    # train the model
    # evaluator.load_epoch(model, 6)
    evaluator.train(model)

    # evaluate mrr for a particular epoch
    evaluator.load_epoch(model, 100)
    # evaluator.load_epoch(model, 31)
    evaluator.get_map(model, evaluate_all=True)
    # for epoch in range(1, 100):
    #     print('Epoch %d' % epoch)
    #     evaluator.load_epoch(model, epoch)
    #     evaluator.get_map(model, evaluate_all=True)
