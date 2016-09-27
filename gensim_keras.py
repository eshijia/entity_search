# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
import re
import codecs
from heapq import nlargest
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
np.random.seed(1337)  # For Reproducibility


# set parameters:
vocab_dim = 300
max_len = 10
n_iterations = 50  # ideally more..
n_exposures = 1
window_size = 1
batch_size = 32
n_epoch = 1
input_length = 10
cpu_count = multiprocessing.cpu_count()
data_locations = {'data/celebrity-test-fenci.txt': 'TEST',
                  'data/celebrity-train-0-fenci.txt': 'TRAIN_NEG',
                  'data/celebrity-train-1-fenci.txt': 'TRAIN_POS',
                  'data/celebrity-all-clean.txt': 'TRAIN_ALL'}


def import_tag(datasets=None):
    ''' Imports the datasets into one of two dictionaries
        Dicts:
            train & test
        Keys:
            values > 12500 are "Negative" in both Dictionaries
    '''
    if datasets is not None:
        train = {}
        test = {}
        corpus = {}
        for k, v in datasets.items():
            with codecs.open(k, encoding='utf-8') as fpath:
                data = fpath.readlines()
            for val, each_line in enumerate(data):
                if v.endswith("NEG") and v.startswith("TRAIN"):
                    train[val] = each_line
                elif v.endswith("POS") and v.startswith("TRAIN"):
                    train[val + 8642] = each_line
                elif v.startswith("TEST"):
                    test[val] = each_line
                else:
                    corpus[val] = each_line
        return train, test, corpus
    else:
        print('Data not found...')


def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [re.split(r'\t|\s+', document.replace('\n', '').strip()) for document in text]
    return text


def create_dictionaries(train=None,
                        test=None,
                        model=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (train is not None) and (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data):
            ''' Words become integers
            '''
            for key in data.keys():
                txt = data[key].replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except KeyError:
                        new_txt.append(0)
                data[key] = new_txt
            return data
        train = parse_dataset(train)
        test = parse_dataset(test)
        return w2indx, w2vec, train, test
    else:
        print('No data provided...')


def make_submit(predict, test_file, submit_file):
    lines = test_file.readlines()
    target_lines = list()
    index = 0
    for line in lines:
        terms = line.strip('\n').split('\t')
        temp = terms[0]
        entities = terms[1:]
        num_candidate = len(entities)

        pr_candidate = [x for x in predict[index:index+num_candidate]]
        index_entities = xrange(num_candidate)
        index_candidates = nlargest(num_candidate, index_entities, key=lambda i: pr_candidate[i])

        for index_candidate in index_candidates:
            temp = temp + '\t' + entities[index_candidate]
        temp += '\n'
        target_lines.append(temp)
        index += num_candidate
    submit_file.writelines(target_lines)

print('Loading Data...')
train, test, corpus = import_tag(datasets=data_locations)
combined = train.values() + test.values()

print('Tokenising...')
combined = tokenizer(combined)
corpus_all = tokenizer(corpus.values())


print('Training a Word2vec model...')
model = Word2Vec(size=vocab_dim,
                 min_count=n_exposures,
                 window=window_size,
                 workers=cpu_count,
                 iter=n_iterations)
model.build_vocab(corpus_all)
model.train(corpus_all)

print('Transform the Data...')
index_dict, word_vectors, train, test = create_dictionaries(train=train,
                                                            test=test,
                                                            model=model)

print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]

print('Creating Datesets...')
X_train = train.values()
y_train = [1 if value >= 8642 else 0 for value in train.keys()]
X_test = test.values()

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert labels to Numpy Sets...')
y_train = np.array(y_train)

print('Defining a Simple Keras Model...')
model = Sequential()  # or Graph or whatever
model.add(Embedding(output_dim=vocab_dim,
                    input_dim=n_symbols,
                    mask_zero=True,
                    weights=[embedding_weights],
                    input_length=input_length))  # Adding Input Length
model.add(LSTM(vocab_dim))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

print('Compiling the Model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epoch)

# print("Evaluate...")
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=batch_size,
#                             show_accuracy=True)
# print('Test score:', score)
# print('Test accuracy:', acc)

result = model.predict(X_test, batch_size=32, verbose=1)
test = codecs.open('data/celebrity.DEVSET.txt', 'r', 'gb18030')
submit = codecs.open('data/celebrity-submit-v6-1.txt', 'w', 'gb18030')
make_submit(result, test, submit)
