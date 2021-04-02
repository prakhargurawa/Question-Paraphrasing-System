# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:50:17 2021

@author: prakh
"""
# Import libraries
import os, sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameter selection
BATCH_SIZE = 64
EPOCHS = 60
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

# Data preprocessing and reading
input_sentences = []
output_sentences = []
output_sentences_inputs = []

# Read training file
df = pd.read_csv('train.csv',header=None)

for i in range(df.shape[0]):
    input_seq = df[0][i][:-1]
    output_seq = df[1][i][:-1]
    output_sentence = output_seq + ' <eos>'
    output_sentence_input = '<sos> ' + output_seq
    input_sentences.append(input_seq)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)
    
print("num samples input:", len(input_sentences))
print("num samples output:", len(output_sentences))
print("num samples output input:", len(output_sentences_inputs))


print(input_sentences[0])
print(output_sentences[0])
print(output_sentences_inputs[0])

# Read eval file
df2 = pd.read_csv('eval.csv',header=None)
eval_inputs = []
for i in range(df2.shape[0]):
    eval_inputs.append(df2[0][i][:-1])

total_input = input_sentences + eval_inputs
print("Total length : ",len(total_input))


# Tokenization and padding
""" It divides a sentence into the corresponding list of word
Then it converts the words to integers """
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
#input_tokenizer.fit_on_texts(input_sentences)
input_tokenizer.fit_on_texts(total_input)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)
print("Ex : ",input_tokenizer.word_index['size'],input_tokenizer.word_index['xbox'],input_tokenizer.word_index['madrasah'])

word2idx_inputs = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)


# Padding
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)
print("encoder_input_sequences[0]:", encoder_input_sequences[0])
print(word2idx_inputs["what"])

decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post') # in the case of the decoder, the post-padding is applied
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)
print("decoder_input_sequences[0]:", decoder_input_sequences[0])

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')


# Word Embeddings
""" With integer reprensentation, a word is represented only with a single integer. 
With vector representation a word is represented by a vector of 50, 100, 200, or 
whatever dimensions you like. Hence, word embeddings capture a lot more information about words. 
Secondly, the single-integer representation doesn't capture the relationships between different words."""
from numpy import array
from numpy import asarray
from numpy import zeros

# Glove vector embeddings
embeddings_dictionary = dict()
glove_file = open(r'glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()


num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
print(embeddings_dictionary["what"])
print(embedding_matrix[539])

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)
# Model
""" output will be a sequence of words, the final shape of the output will be:
(number of inputs, length of the output sentence, the number of words in the output)"""
decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)
print(decoder_targets_one_hot.shape)


# the final layer of the model will be a dense layer, therefore we need the outputs in the form of one-hot encoded vectors,
# since we will be using softmax activation function at the dense layer.
for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

# Stacked encoder-decoder
# Encoder
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder1 = LSTM(LSTM_NODES, return_state= True, return_sequences=True)
encoder = LSTM(LSTM_NODES, return_state=True)
encoder_outputs, h, c = encoder(encoder1(x))
encoder_states = [h, c]

# Decoder
decoder_inputs_placeholder = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder(decoder_inputs_x, initial_state=encoder_states))
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)  
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
)



encoder_model = Model(encoder_inputs_placeholder, encoder_states)
decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
decoder_outputs, state_h, state_c = decoder(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_outputs2, h, c = decoder_lstm(decoder(decoder_inputs_single_x, initial_state=[state_h, state_c]))
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}


def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

# Selecting random string from train
i = np.random.choice(len(input_sentences))
input_seq = encoder_input_sequences[i:i+1]
translation = translate_sentence(input_seq)
print('-')
print('Input:', input_sentences[i])
print('Response:', translation)

"""
word2idx_inputs["was"]
word2idx_inputs["the"]
word2idx_inputs["landing"]
word2idx_inputs["successful"]
"""

def encode_to_input(s):
    x  = []
    for w in s.split(" "):
        x.append(word2idx_inputs[w.lower()])
    return pad_sequences([x], maxlen=max_input_len)
    
translation = translate_sentence(encode_to_input("What kind of glass exists in nature"))
print('Response:', translation)
# Response: what in mongol muslim ii

translation = translate_sentence(encode_to_input("was the landing successful"))
print('Response:', translation)
# Response: of 2005

translation = translate_sentence(encode_to_input("what county is raleigh in"))
print('Response:', translation)
# Response: of mongol green colony



"""
Reference : 
    1. https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
    2. https://stackoverflow.com/questions/52465971/keras-seq2seq-stacked-layers (for stacking)
    3. https://arxiv.org/pdf/1610.03098.pdf (Similiar paper: Neural Paraphrase Generation with Stacked Residual LSTM Networks)
"""


