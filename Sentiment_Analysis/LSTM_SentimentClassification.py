'''

NaiveBayes sentiment classifciation model

'''
#import stuff to allow code ot run on python2 & 3
from __future__ import (absolute_import, division, print_function)
from six.moves import urllib

#useful libraries
import collections
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import random
import re
import tarfile
import tensorflow as tf

DOWNLOADED_FILENAME = 'ImdbReviews.tar.gz'

def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretreive(url_path, DOWNLOADED_FILENAME)

    print("Downloaded File")
    
TOKEN_REGEX = re.compile("[^-Za-z0-9 ]+")

def get_reviews(dirname, positive=True):
    
    label = 1 if positive else 0
    
    reviews = []
    labels = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            
            with open(dirname + filename, 'r+', encoding="utf8") as f:
                review = f.read() #.decode('utf-8')
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)
                
                reviews.append(review)
                labels.append(label)
        
    return reviews, labels
    
def extract_labels_data():
    
    # if file has not already been extracted
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            tar.extractall()
            tar.close
    
    positive_reviews, positive_labels = get_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews, negative_labels = get_reviews("aclImdb/train/neg/", positive=False)
    
    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels
    
    return labels, data

URL_PATH = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

download_file(URL_PATH)

labels, data = extract_labels_data()

max_document_length = max([len(x.split(" ")) for x in data])
print("Max doc length is: ", max_document_length)

MAX_SEQUENCE_LENGTH = 250 
#pad short and truncate longer
#depracated - needs replacing
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)
print("Vocab maybe: ", vocab_processor)
print("Vocab: ", vocab_processor.vocabulary_)

x_data = np.array(list(vocab_processor.fit_transform(data)))
y_output = np.array(labels)

vocabulary_size = len(vocab_processor.vocabulary_)
print("Vocab size: ", vocabulary_size)

#shuffle data
np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]

TRAIN_DATA = 5000
TOTAl_DATA = 6000

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAl_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAl_DATA]

tf.reset_default_graph()

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32, [None])

NUM_EPOCHS = 20
BATCH_SIZE = 25
EMBEDDING_SIZE = 50
MAX_LABEL = 2

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_SIZE], -1.0, 1.0))

embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

lstmCell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE)

#dropout wrapper to avoid overfitting
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

#output, (final_state, other_state_info)
_, (encoding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

#softmax is part of loss function fro some reason???
logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

loss = tf.reduce_mean(cross_entropy)

prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y,tf.int64))

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    init.run()
    
    for epoch in range(NUM_EPOCHS):
        
        num_batches = int(len(train_data) // BATCH_SIZE) + 1
        
        for i in range(num_batches):
        
            min_ix = i * BATCH_SIZE
            max_ix = np.min([len(train_data), ((i+1) * BATCH_SIZE)])
            
            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y: y_train_batch}
            sess.run(train_step, feed_dict=train_dict)
            
            train_loxx, train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
            
        test_dict = {x: test_data, y: test_target}
        
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict=test_dict)

        print('Epoch: {}, Test Loss: {:2}, Test Acc:{:5}'.format(epoch +1, test_loss, test_acc))
























