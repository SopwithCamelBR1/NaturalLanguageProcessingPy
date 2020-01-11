'''

NaiveBayes sentiment classifciation model

'''
#import stuff to allow code ot run on python2 & 3
from __future__ import (absolute_import, division, print_function)
from six.moves import urllib

#useful libraries
import nltk
import os
import re
import tarfile
import numpy as np

debug = 1

'''for IMDB reviews'''
def download_file(url_path):
    
    # check if file already downloaded
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)
        
    print("downloaded file")

def get_imdb_reviews(dirname, positive=True):
    
    label = 1 if positive else 0
    
    reviews = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            
            with open(dirname + filename, 'r+', encoding="utf8") as f:
                review = f.read() #.decode('utf-8')
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)
                
                reviews.append((review, label))
        
    return reviews
    
def extract_imdb_reviews():
    
    # if file has not already been extracted
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            tar.extractall()
            tar.close
    
    positive_reviews = get_imdb_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews = get_imdb_reviews("aclImdb/train/neg/", positive=False)
    
    return positive_reviews, negative_reviews
    
'''for local reviews'''
def get_reviews(path, positive=True):
    label = 1 if positive else 0

    with open(path, 'r') as f:
        review_text = f.readlines()
    
    reviews = []
    for text in review_text:
        #return tuple of review text and pos/neg label
        reviews.append((text, label))
        
    return reviews

def extract_reviews(path_pos, path_neg):
    positive_reviews = get_reviews(path_pos, positive=True)
    negative_reviews = get_reviews(path_neg, positive=False)
    
    len_pos, len_neg = len(positive_reviews), len(negative_reviews)
    
    return positive_reviews, negative_reviews, len_pos, len_neg

def get_vocabulary(train_dataset):
    words_set = set()
    
    for review in train_dataset:
        words_set.update(review[0].split())
        
    return list(words_set)

def extract_features(review_text):
    
    #Split the review into words and create a set of words
    review_words = set(review_text.split())
    
    features = {}
    for word in vocabulary:
        features[word] = (word in review_words)
    
    return features

def sentiment_calculator(review_text, classifier):
    
    features = extract_features(review_text)
    
    return classifier.classify(features)
    
def classify_reviews(test_positive_reviews, test_negative_reviews, classifier):
    
    positive_results = [sentiment_calculator(review[0], classifier) for review in test_positive_reviews]
    negative_results = [sentiment_calculator(review[0], classifier) for review in test_negative_reviews]
    
    true_positives = sum(x > 0 for x in positive_results)
    true_negatives = sum(x == 0 for x in negative_results)
    total_accurate = true_positives + true_negatives
    
    percentage_true_positives = ( float(true_positives) / len(positive_results) ) * 100
    percentage_true_negatives = ( float(true_negatives) / len(negative_results) ) * 100
    accuracy = ( float(total_accurate) / ( len(positive_results)+len(negative_results) ) ) * 100
    
    return true_positives, true_negatives, total_accurate, percentage_true_positives, percentage_true_negatives, accuracy

 
if __name__ == "__main__":
    
    LOCAL = True
    
    if LOCAL == True:
        
        #Paths to positive and negatvie reviews
        PATH_POS = "rt-polaritydata/rt-polarity.pos"
        PATH_NEG = "rt-polaritydata/rt-polarity.neg"    

        positive_reviews, negative_reviews, len_pos, len_neg = extract_reviews(PATH_POS, PATH_NEG)

        TRAIN_DATA = 5000
        TOTAL_DATA = max(len_pos, len_neg)

        train_reviews = positive_reviews[:TRAIN_DATA] + negative_reviews[:TRAIN_DATA]
        test_positive_reviews = positive_reviews[TRAIN_DATA:TOTAL_DATA]
        test_negative_reviews = negative_reviews[TRAIN_DATA:TOTAL_DATA]

        vocabulary = get_vocabulary(train_reviews)

        if debug >=1:
            print("No. of pos reviews: ", len_pos)
            print("No. of neg reviews: ", len_neg)
            print("Total data is: ,", TOTAL_DATA)
            print("First ten wrods in vocabulary: ", vocabulary[:10])

        train_features = nltk.classify.apply_features(extract_features, train_reviews)

        trained_classifier = nltk.NaiveBayesClassifier.train(train_features)

        TEST_REVIEW = "What an amazing movie!"
        sentiment = sentiment_calculator(TEST_REVIEW, trained_classifier)
        print("Sentiment of '", TEST_REVIEW, "': ", sentiment)

        true_positives, true_negatives, total_accurate, percentage_true_positives, percentage_true_negatives, accuracy = classify_reviews(test_positive_reviews, test_negative_reviews, trained_classifier)

        print("Positive Accuracy: ", percentage_true_positives, "% (", true_positives, ")")
        print("Negatives Accuracy: ", percentage_true_negatives, "% (", true_negatives, ")")
        print("Overall Accuracy: ", accuracy, "% (", total_accurate, ")")

    else:
        URL_PATH = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'   
        DOWNLOADED_FILENAME = 'ImdbReviews.tar.gz'
        #regex to clean up review - this removes all special characters
        TOKEN_REGEX = re.compile("[^-Za-z0-9 ]+")

        download_file(URL_PATH)

        positive_reviews, negative_reviews= extract_imdb_reviews()

        TRAIN_DATA = 5000
        TOTAL_DATA = 6000

        train_reviews = positive_reviews[:TRAIN_DATA] + negative_reviews[:TRAIN_DATA]
        test_positive_reviews = positive_reviews[TRAIN_DATA:TOTAL_DATA]
        test_negative_reviews = negative_reviews[TRAIN_DATA:TOTAL_DATA]

        vocabulary = get_vocabulary(train_reviews)

        if debug >=1:
            print("No. of pos reviews: ", len(positive_reviews))
            print("No. of neg reviews: ", len(negative_reviews))
            print("Total data is: ,", TOTAL_DATA)
            print("First ten words in vocabulary: ", vocabulary[:10])

        train_features = nltk.classify.apply_features(extract_features, train_reviews)

        trained_classifier = nltk.NaiveBayesClassifier.train(train_features)

        TEST_REVIEW = "What an amazing movie!"
        sentiment = sentiment_calculator(TEST_REVIEW, trained_classifier)
        print("Sentiment of '", TEST_REVIEW, "': ", sentiment)

        true_positives, true_negatives, total_accurate, percentage_true_positives, percentage_true_negatives, accuracy = classify_reviews(test_positive_reviews, test_negative_reviews, trained_classifier)

        print("Positive Accuracy: ", percentage_true_positives, "% (", true_positives, ")")
        print("Negatives Accuracy: ", percentage_true_negatives, "% (", true_negatives, ")")
        print("Overall Accuracy: ", accuracy, "% (", total_accurate, ")")


    

    









