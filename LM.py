
# coding: utf-8

# In[1]:

import argparse


import spacy
import numpy as np

import pickle
import json
import os
import csv
import pprint as pp

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from random import shuffle, choice, sample

from sklearn.model_selection import StratifiedShuffleSplit

from copy import copy

import warnings
warnings.filterwarnings('ignore')

data_path = ''

nlp = spacy.load('en_core_web_sm')

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl


sns.set(color_codes=True)

import warnings
warnings.filterwarnings('ignore')




# In[2]:


from keras.models import Model, Sequential
from keras.layers import Input, CuDNNLSTM, Dense, LSTM
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Merge, Dot, Concatenate, Flatten, Permute, Multiply, dot, concatenate
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import load_model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from collections import Counter
# In[3]:

spam_dataset = pickle.load(open(os.path.join(data_path, 'tweets_together.pkl'), 'rb'))

print(len(spam_dataset))


# In[4]:


parser = argparse.ArgumentParser(description='language model')
parser.add_argument('--m', type=int, dest='maxlen', help='max characters to give for predicting next char', default=5)
parser.add_argument('--t', type=int, dest='train', help='boolTrain', default=1, choices=[0,1])
parser.add_argument('--f', type=str, dest='file', help='loadfile', default='default_file')
parser.add_argument('--p', type=int, dest='n_predict', help='nb of predicted tweets', default=5)
parser.add_argument('--r', type=int, dest='warm_up', help='random warm up', default=1, choices=[0,1])

args = parser.parse_args()



spam_dataset = pickle.load(open(os.path.join(data_path, 'tweets_together.pkl'), 'rb'))
tokenized = [list(x) for x in spam_dataset]

start_token = [s[1] for s in tokenized if len(s) > 1]
len(start_token)

maxlen = max([len(x) for x in tokenized])
avglen = sum([len(x) for x in tokenized])/len(tokenized)
print(maxlen, avglen)


vocab = [t for s in spam_dataset for t in s]
print('num tokens: ', len(vocab))
vocab_counter = Counter(vocab)
vocab = [w for w, v in vocab_counter.items() if v>2]
vocab = ['<PAD>','<UNK>', '<SOS>', '<EOS>'] + vocab
nb_vocab = len(vocab)
print('Num de features a usar: ', nb_vocab)


w2id = {k:i for i, k in enumerate(vocab)}
id2w = {i:k for k, i in w2id.items()}

maxlen = min(maxlen, args.maxlen)

init_chars = [x[:maxlen] for x in tokenized]
for i in range(len(init_chars)):
    tmp = init_chars[i]
    tmp.insert(0, '<SOS>')
    init_chars[i] = tmp[:maxlen]


step = 1
data_train = []
for x in tokenized:
    x.insert(0, '<SOS>')
    x.append('<EOS>')
    for i in range(0, len(x) - maxlen, step):
        data_train.append((x[i: i + maxlen], x[i + maxlen]))

print('nb sequences:', len(data_train))


# In[13]:


SAMPLE_EVERY = 10
PLOT_EVERY = 1


# In[14]:


def sample_pred(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[25]:


class Sampletest(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % SAMPLE_EVERY == 0  and epoch>0:
            data_test = []
            nb_samples = 1

            params = {
                'maxlen': maxlen,
                'vocab': nb_vocab,
                'use_embeddings': True
                }
            for _ in range(nb_samples):
                data_test = choice(init_chars)
                x_pred = np.zeros((1, params['maxlen'], params['vocab']), dtype=np.bool)
                for diversity in [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]:
                    print('----- diversity:', diversity)
                    sentence = copy(data_test)
                    generated = copy(data_test)
                    for i in range(len(data_test), 140):
                        x_pred = np.zeros((1, params['maxlen'], params['vocab']))
                        for t, char in enumerate(sentence):
                            x_pred[0, t, w2id[char] if char in w2id else w2id['<UNK>']] = 1.
                        preds = self.model.predict(x_pred, verbose=0)[0]
                        next_index = sample_pred(preds, diversity)
                        next_char = id2w[next_index]
                        if next_char == '<EOS>':
                            break
                        generated += [next_char]
                        sentence = sentence[1:]
                        sentence += [next_char]
                    print(''.join(generated))


# In[16]:


class HistoryDisplay(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.epochs = []
        self.fig, self.ax = plt.subplots()
        #plt.show()

        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.accs.append(logs['acc'])
        if epoch % PLOT_EVERY == 0:

            self.ax.clear()
            self.ax.plot(self.epochs, self.accs, 'g', label='acc')
            self.ax.plot(self.epochs, self.losses, 'b', label='loss')
            legend = self.ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            #display.clear_output(wait=True)
            #display.display(pl.gcf())
            self.fig.canvas.draw()

            #plt.draw()



# In[17]:


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# In[18]:


class LM:
    def __init__(self, **kwargs):
        self.params = kwargs.pop('params', None)

    def compile_bidirectional(self, params={}):


        encoder_inputs = Input(shape=(params['maxlen'], params['vocab']), name='encoder_input')
        encoder_embedding = encoder_inputs

        #print(encoder_embedding.shape)

        lstm = CuDNNLSTM(params['rnn_hidden_size'], return_sequences=True, name='rnn_1')
        if 'bidirectional' in params and params['bidirectional']:
            encoder_outputs = Bidirectional(lstm)(encoder_embedding)
            lstm2 = CuDNNLSTM(params['rnn_hidden_size'], return_sequences=False, name='rnn_1')
            encoder_outputs = Bidirectional(lstm2)(encoder_outputs)
        else:
            encoder_outputs = lstm(encoder_embedding)

        decoder_outputs = Dense(params['vocab'], activation="softmax")(encoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model(encoder_inputs, decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, model, data, params={}):

        callbacks = self._get_callbacks()
        print(callbacks)
        if 'shuffle' in params and params['shuffle']:
            shuffle(data)
        print(data[0])
        sentences, next_chars = zip(*data)
        x = np.zeros((len(data), params['maxlen'], params['vocab']), dtype=np.bool)
        y = np.zeros((len(data), params['vocab']), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, w2id[char] if char in w2id else w2id['<UNK>']] = 1
            y[i, w2id[next_chars[i]] if next_chars[i] in w2id else w2id['<UNK>']] = 1
        model.fit(x, y,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        callbacks=callbacks,
                 verbose=1)

    def predict(self, model, data, params={}):
        """Short summary.

        Parameters
        ----------
        model : type
            Description of parameter `model`.
        data : type
            Description of parameter `data`.
        params : type
            Description of parameter `params`.

        Returns
        -------
        type
            Description of returned object.

        """
        x_pred = np.zeros((1, params['maxlen'], params['vocab']), dtype=np.bool)
        y_pred = []
        for diversity in [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]:
            sentence = copy(data)
            generated = copy(data)
            for i in range(len(data), 200):
                x_pred = np.zeros((1, params['maxlen'], params['vocab']))
                for t, char in enumerate(sentence):
                    x_pred[0, t, w2id[char] if char in w2id else w2id['<UNK>']] = 1.
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample_pred(preds, diversity)
                next_char = id2w[next_index]
                if next_char == '<EOS>':
                    break
                generated += [next_char]
                sentence = sentence[1:]
                sentence += [next_char]
            y_pred.append(''.join(generated))
        return y_pred

    def load(self, model_path='lminesymayordomo_7.h5'):
        print('MODEL PATH: ', model_path)
        return load_model(model_path)

    def _get_callbacks(self, model_path='lm'):
        model_path = '{}.h5'.format(args.file)
        es = EarlyStopping(monitor='loss', patience=4, mode='auto', verbose=0)
        save_best = ModelCheckpoint(model_path, monitor='loss', verbose = 0, save_best_only=True, save_weights_only=False, period=2)
        st = Sampletest()
        rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=0)
        return [st, save_best]


# In[26]:





# In[20]:


LOAD_MODEL = False
bTrain = True


# In[27]:


lm = LM()
if args.train:
    compile_params = {
        'vocab': nb_vocab,
        'rnn_hidden_size': 512,
        'maxlen': maxlen,
        'use_embeddings': True,
        'bidirectional': True,
        'dropout': 0.6,
        'rnn_dropout': 0.6
    }
    pp.pprint(compile_params)
    compile_params['w2id'] = w2id
    compile_params['id2w'] = id2w
    pickle.dump(compile_params, open('{}_config.pkl'.format(args.file), 'wb'))
    lm_model = lm.compile_bidirectional(params=compile_params)
else:
    model_path = '{}.h5'.format(args.file)
    lm_model = lm.load(model_path=model_path)
    compile_params = pickle.load(open('{}_config.pkl'.format(args.file), 'rb'))
    w2id = compile_params.pop('w2id')
    id2w = compile_params.pop('id2w')
    pp.pprint(compile_params)
    lm_model.summary()



# In[ ]:


if args.train:
    train_params = {
        'epochs': 300,
        'batch_size': 128,
        'shuffle': True,
        'vocab': nb_vocab,
        'maxlen': maxlen,
        'use_embeddings': True
    }
    pp.pprint(train_params)
    lm.train(model=lm_model, data=data_train, params=train_params)


fpath = '{}_samples.txt'.format(args.file)
print(fpath)
f = open(fpath, 'w')

for _ in range(args.n_predict):
    if args.warm_up:
        data_pred = choice(init_chars)
    else:
        data_pred = input('warm up chars (Max used chars {})'.format(compile_params['maxlen']))
        data_pred = list(data_pred)[:compile_params['maxlen']]
    y_pred = lm.predict(model=lm_model, data=data_pred, params=compile_params)
    for s in y_pred:
        f.write('\n{}\n'.format(s))

f.close()
