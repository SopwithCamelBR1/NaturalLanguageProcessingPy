'''
Extraction-based text summarization programme.
Takes a URL as an input

Required Installs:
pip install bs4
pip install lxml
pip install nltk - "nltk.download('all')" required for the first run (this downloads all, not sure which exact libraries needed for this programme)
'''
import sys
import bs4 as bs  
import urllib.request  
import re
import nltk
#ltk.download('all')
import heapq

#number of sentences to extract
no_sentences=20

'''
Data Importing
'''
#scrape
#url='https://en.wikipedia.org/wiki/Artificial_intelligence'
url=sys.argv[1]
scraped_data = urllib.request.urlopen(url)  
article = scraped_data.read()

#parse
parsed_article = bs.BeautifulSoup(article,'lxml')

'''
Data Pre-processing
'''
#splitting text into paragraphs
paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:  
    article_text += p.text

# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
article_text = re.sub(r'\s+', ' ', article_text)

# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

#sentence tokenization
sentence_list = nltk.sent_tokenize(article_text)

#Stopwords
stopwords = nltk.corpus.stopwords.words('english')

#Frequency of Occurance
word_frequencies = {}  
for word in nltk.word_tokenize(formatted_article_text):  
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

#Weighted Frequency
maximum_frequncy = max(word_frequencies.values())
for word in word_frequencies.keys():  
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)           

#Sentence Scores
sentence_scores = {}  
for sent in sentence_list:  
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
#Summary
summary_sentences = heapq.nlargest(no_sentences, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)  
print(summary) 