'''

This script demonstrates how to implement a basic character-level
seq2seq model. We apply it to translating short English sentenses into
short French sentences, character-by-character. 
Note that it is fairly unusual to do character-level machine translation,
as word-level models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain and corresponding target sequences
    from another domain(Eng->Fre)
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into the
    same sequence but offset by one timestep in the future,
'''

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

import ipdb                  # For debug

batch_size = 64              # Batch size for training.
epochs = 100                 # Number of epochs to train for.
latent_dim = 256             # Latent dimensionality of the encoding space.
num_samples = 10000          # Number of samples to train on.
data_path = 'fra-eng/fra.txt' # Path to the data txt file on disk.

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)

input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
        (len(input_texts

ipdb.set_trace()

