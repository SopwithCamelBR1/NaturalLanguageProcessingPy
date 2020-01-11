'''
Word2Vec tensorflow model

Skipgram

perhaps try to use sigmoids? but would need to normalise inputs(i.e. frequencies) to between 0-1
'''

#import stuff to allow code ot run on python2 & 3
from __future__ import (absolute_import, division, print_function)
from six.moves import (urllib, xrange)
#allows us to slice dictionaries (only used in printing debug information)
from itertools import islice

#useful libraries
import collections
import math
import os
import nltk
import random
import sys
import zipfile

import numpy as np
import tensorflow as tf

print("It's Alive!")

debug = 0

#Attempt to download file and check it 
def maybe_downloaded(downloaded_filename, url_path, expected_bytes):
    '''Checks if file exists, and if not attempts download. Size of file is then checked.
    *
    * Args:
    *   downloaded_filename - zip file with a single text file in it.
    *   url_path - url to attempt download from
    *   expected_bytes - expected size of file, used to check it
    *
    * Returns:
    *   null
    '''
    if not os.path.exists(downloaded_filename):
        try:
            filename, _ = urllib.request.urlretrieve(url_path, downloaded_filename)
        except urllib.error.URLError:
            print('Failed to open URL: ', url_path)
            sys.exit()
    else:
        print('File ', downloaded_filename, 'present locally')        
    
    #Check if size of file mis what was expected
    statinfo = os.stat(downloaded_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified file from this path: ', url_path)
        print('Downlaoded file: ', downloaded_filename)
    else:
        print('Unexpected file size ', downloaded_filename)
        print('Expected : ', expected_bytes)
        print('Received: ', statinfo.st_size)
  
      
#Create array of words from the documents        
def read_words(downloaded_filename):
    '''Takes downloaded file and turns it into an array of words. 
    *
    * Args:
    *   downloaded_filename - zip file with a single text file in it.
    *
    * Returns:
    *   words - array of words (i.e. ["this", "is", "array", "of", "words"])
    '''
    with zipfile.ZipFile(downloaded_filename) as f:
        firstfile = f.namelist()[0]
        filestring = tf.compat.as_str(f.read(firstfile))
        words = filestring.split()
    
    #debugging
    if debug >= 1:
        print('File read and processed word array created')
    if debug >= 2:    
        print('First ten in words array are: ', words[:10])
    
    return words

#build the datasets    
def build_dataset(words, n_words):
    '''Takes an Array of words and creates datasets of the n most frequent words.
    *
    * Args:
    *   words - array of words (i.e. ["this", "is", "array", "of", "words"])
    *   n_words - number of words to use (i.e. use n_word most frequent words) 
    *
    * Returns:
    *   word_counts - [[WORD : FREQUENCY]], mapping the word to the number of times it appears in the dataset
    *   word-indexes - [INDEX], list of all words from original dataset in index form
    *   dictionary - [[WORD : INDEX]]
    *   reversed_dictionary - [[INDEX : WORD]]
    '''
    word_counts = [['UNKOWN', -1]]
    
    counter = collections.Counter(words)
    word_counts.extend(counter.most_common(n_words - 1))
    
    dictionary = dict()
    
    for word, _ in word_counts:
        dictionary[word] = len(dictionary)
        
    word_indexes = list()
    
    unkown_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 # dictionary['UNKOWN']
            unkown_count += 1
        
        word_indexes.append(index)
        
    word_counts[0][1] = unkown_count
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    # Debugging
    if debug >= 1:
        print('Datasets built')
    if debug >= 2:    
        print('word_counts is: ', word_counts[:10])
        print('word_indexes is: ', word_indexes[:10])
        #this method for printing dictionary only seems to print the value not the key
        print('dictionary is: ', list(islice(dictionary, 10)))
        print('reversed_dictionary is: ', list(islice(reversed_dictionary, 10)))
    
    return word_counts, word_indexes, dictionary, reversed_dictionary
    
#Generate a Batch of data from the datasets created above.
def generate_batch(word_indexes, batch_size, num_skips, skip_window):
    ''' Creates batches of data to be used in training 
    *
    * Args:
    *   word_indexes - the array of word indexes created in build_datset()
    *   batch_size - size of batch
    *   num_skips - ??????sometihng????
    *   skip_window - the number of words either side of the input word (i.e. context window)
    *
    * Returns:
    *   batch - 
    *   labels - 
    '''
    global global_index
    
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1 # [skip_window - input_word - skip_window]
    
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)
        
    for i in range (batch_size // num_skips):
        target = skip_window # input word at the center of the buffer
        targets_to_avoid = [skip_window]
    
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
                
            targets_to_avoid.append(target)
    
            batch[i * num_skips + j] = buffer[skip_window] #this is the input word
            labels[i * num_skips + j, 0] = buffer[target] # these are the context words
        
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)
        
    global_index = (global_index + len(word_indexes) - span) % len(word_indexes)
    
    # Debugging
    if debug >= 1:
        print('Batch and Labels built')
    
    return batch, labels

#Model
def create_model(batch_size, embedding_size, valid_examples, vocabulary_size):
    ''' Creates single  hidden linear layer model
    *
    '''
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
                                    
    bias = tf.Variable(tf.zeros([vocabulary_size]))

    hidden_out = tf.matmul(embed, tf.transpose(weights)) + bias
    
    return hidden_out, train_inputs, train_labels, valid_dataset, embeddings, embed, weights, bias

#Predictor - either Softmax (default) or NCE
def calc_loss(model_output, target, *argv, predictor='softmax'):
    ''' returns the loss, using different predictrs
    *
    * currently two possible predictors:
    *   Softmax:
    *       Args: model_output, target, number_of_outputs(vocab size), predictor='softmax'
    *   Noise Contrastive Estimator:
    *       Args: model_output, target, weights_to_train, biases_to_train, model_inputs, number_of_outputs_to_sampe, number_of_outputs????, predictor='nce'
    '''
    if predictor == 'softmax':
        train_one_hot = tf.one_hot(target, argv[0])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_out, labels=train_one_hot))
    elif predictor == 'nce':
        #assert length(argv) == 5
        loss = tf.reduce_mean(tf.nn.nce_loss(labels=target, weights=argv[0], biases=argv[1], inputs=argv[2], num_sampled=argv[3], num_classes=argv[4]))  
    else:
        print('Predictor not recognised')
        sys(exit)
    
    return loss  
    
   
#Cosine differences or something
def calc_similarity(embeddings, valid_dataset):
    ''' finds similar words, not sure what this does
    * 
    '''
    l2_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    normalized_embeddings = embeddings / l2_norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    return similarity

'''
* User DEFINED VARIABLES
'''
   
#Variables to use for file download
DOWNLOADED_FILENAME = 'SampleText.zip'
URL_PATH = 'http://mattmahoney.net/dc/text8.zip'
FILESIZE = 31344016
VOCABULARY_SIZE = 5000

#Global index into words maintained across batches
global_index=0

#batch variables
BATCH_SIZE = 128
EMBEDDING_SIZE = 50
SKIP_WINDOW = 2
NUM_SKIPS = 2

#check if words are similar/valid???????
VALID_SIZE = 16
VALID_WINDOW = 100
valid_examples = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)

#for Noise Contrast Estimator
NUM_SAMPLES = 64

#Learning varialbes
LEARNING_RATE = 0.1
NUM_STEPS = 200001

'''
* Running data processing functions
'''
maybe_downloaded(DOWNLOADED_FILENAME, URL_PATH, FILESIZE)
vocabulary = read_words(DOWNLOADED_FILENAME)
word_counts, word_indexes, dictionary, reversed_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)
#print outputs for debugging
if debug >= 1:    
    print('Vocabulary is: ', vocabulary[:10])
    print('word_counts is: ', word_counts[:10])
    print('word_indexes is: ', word_indexes[:10])
    #this method for printing dictionary only seems to print the value not the key
    print('dictionary is: ', list(islice(dictionary, 10)))
    print('reversed_dictionary is: ', list(islice(reversed_dictionary, 10)))

del vocabulary #this is no longer needed so can be cleaned up.
    
'''
* Running model & loss functions
'''
tf.reset_default_graph()

model_out, train_inputs, train_labels, valid_dataset, embeddings, embed, weights, bias = create_model(BATCH_SIZE, EMBEDDING_SIZE, valid_examples, VOCABULARY_SIZE)

#loss = calc_loss(model_out, train_labels, VOCABULARY_SIZE, predictor='softmax')
loss = calc_loss(model_out, train_labels, weights, bias, embed, NUM_SAMPLES, VOCABULARY_SIZE, predictor='nce')

similarity = calc_similarity(embeddings, valid_dataset)

'''
* Running and Training the Model
'''
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    total_loss = 0
    for step in xrange(NUM_STEPS):
        batch_inputs, batch_labels = generate_batch(word_indexes, BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
        
        feed_dict = {train_inputs: batch_inputs, train_labels:batch_labels}
        
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        total_loss += loss_val
        
        #print average loss every 2000 steps
        if step % 2000 == 0:
            average_loss = 0 # initialise it so print function can use it - this shouldn't be necessary...
            if step > 0:
                average_loss = total_loss / step
                
            print('Average loss at step ', step, ': ', average_loss)
            
        #evaluate closest words?
        
        if step % 10000 == 0:
            sim = similarity.eval() #basically session.run(similarity)
            
            for i in xrange(VALID_SIZE):
                
                valid_word = reversed_dictionary[valid_examples[i]]
                
                top_k = 8 #number of nearest neighbour
                
                nearest = (-sim[i, :]).argsort()[1:top_k +1]
                log_str = 'Nearest to %s:' % valid_word
                               
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
            print('\n')
        
        
'''
* END OF CODE
'''