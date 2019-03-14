import os
import numpy

import tensorflow as tf

# DEBUG
import ipdb

DATASET_ROOT = '/home/jglee/Research/datasets/LanguageTranslationDataset/europarl/training'
READ_SIZE = NUM_INST = 50
NUM_UNITS = 100

fr = open(os.path.join(DATASET_ROOT, 'europarl-v7.fr-en.en'), 'r')
en_strs = []
en_vocab = set()
for _ in range(READ_SIZE):
    en_str = fr.readline().split('\n')[0]
    en_vocab = en_vocab.union(set(list(map(ord, en_str))))
    en_strs.append(en_str)
fr.close()

fr = open(os.path.join(DATASET_ROOT, 'europarl-v7.fr-en.fr'), 'r')
fr_strs = []
fr_vocab = set()
for _ in range(READ_SIZE):
    fr_str = fr.readline().split('\n')[0]
    fr_vocab = fr_vocab.union(set(list(map(ord, fr_str))))
    fr_strs.append(fr_str)
fr.close()

buckets = []
for i in range(NUM_INST):
    buckets.append((len(en_strs[i])+1, len(fr_strs[i])+1))

encoder_inputs = []
decoder_inputs = []

for i in range(buckets[0][0]):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))

for i in range(buckets[0][1] + 1):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))

targets = decoder_inputs[:1]

weight_x = tf.get_variable("weight_x", [NUM_UNITS])
weight_h = tf.get_variable("weight_h", [NUM_UNITS])
#hidden_t = tf.matmul(weight_x, encoder_inputs[0]) + tf.matmul(weight_h, 

ipdb.set_trace()
