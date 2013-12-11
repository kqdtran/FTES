# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Connects to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBAMkOPptLZCUoAlRsUfD5nOnLcDo32ce6KZBKCxhOJ6E4YBjrQAFAxmoWmIdyd8AWl1TJoghswmnUKrTixUB5Toil9FsyfTy8YLjCfq292flRfYIJML0nzf3QwpR5nOnOGXLfntTlvriHnrXZBRQUChXMv3NPNbPTdeoKdgxDuXt7QCpfhVO8GCUXcNZAggZDZD'

# <markdowncell>

# # Import packages and dependencies

# <codecell>

import facebook  # pip install facebook-sdk, not facebook
import os
import random

# Plotting
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
%matplotlib inline

# Making request, pretty print stuff
import requests
import json
import simplejson as sj

# NLP!
import string
import nltk
from nltk.corpus import stopwords
import tagger as tag
import enchant
from nltk.metrics import edit_distance
from pattern.vector import Document, Model, TFIDF, LEMMA

# <markdowncell>

# ## Builds some NLP tools, including a lemmatizer, stemmer, chunker, etc.

# <codecell>

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
table = string.maketrans("", "")

# Noun phrase chunker
grammar = r"""
    # Nouns and Adjectives, terminated with Nouns
    NBAR:
        {<NN.*|JJ>*<NN.*>}
        
    # Above, connected with preposition or subordinating conjunction (in, of, etc...)
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}"""
chunker = nltk.RegexpParser(grammar)

# POS tagger
tagger = tag.tagger()

# Spelling corrector
spell_dict = enchant.Dict('en')

# <markdowncell>

# ## Helper functions for pretty printing, converting text to ascii, etc.

# <codecell>

def pp(o):
    '''
    A helper function to pretty-print Python objects as JSON
    '''
    print json.dumps(o, indent=2)
    
def to_ascii(unicode_text):
    '''
    Converts unicode text to ascii. Also removes newline \n \r characters
    '''
    return unicode_text.encode('ascii', 'ignore').\
            replace('\n', ' ').replace('\r', '').strip()
    
def strip(s, punc=False):
    '''
    Strips punctuation and whitespace from a string
    '''
    if punc:
        stripped = s.strip().translate(table, string.punctuation)
        return ' '.join(stripped.split())
    else:
        return ' '.join(s.strip().split())
    
def tokenize(s):
    '''
    Tokenizes a string
    '''
    return nltk.tokenize.wordpunct_tokenize(s)

def lower(word):
    '''
    Lowercases a word
    '''
    return word.lower()

def lemmatize(word):
    '''
    Lemmatizes a word
    '''
    return lemmatizer.lemmatize(word)

def stem(word):
    '''
    Stems a word using the Porter Stemmer
    '''
    return stemmer.stem_word(word)

def correct(word, max_dist=2):
    '''
    Corrects spelling of a word. If the word is already correct,  cannot be corrected at all,
    or the edit distance is greater than 2, just returns the input word
    
    If it can be corrected, finds a list of suggestions and 
    returns the first result whose edit distance doesn't exceed a threshold
    
    Edit distance: https://en.wikipedia.org/wiki/Edit_distance
    '''
    if spell_dict.check(word):
        return word
    
    suggestions = spell_dict.suggest(word)
    if suggestions and edit_distance(word, suggestions[0]) <= max_dist:
        return suggestions[0]
    return word

# <markdowncell>

# ### Some tiny examples

# <codecell>

# Works
print correct('aninal')
print correct('behols'), '\n'

# Doesn't work... very well
print correct('aisplane') # airplane
print correct('facebook') # don't separate them...

# <codecell>

# Creates a connection to the Graph API with your access token
g = facebook.GraphAPI(ACCESS_TOKEN)

# Sets a few request URL for later use
me_url = 'https://graph.facebook.com/me/home?q=facebook&access_token=' + ACCESS_TOKEN
req = requests.get(me_url)
req

# <codecell>

my_feed = sj.loads(req.content)['data']
print len(my_feed)

# <markdowncell>

# # Retrieves a group's feed

# <markdowncell>

# **To retrieve a group's feed, you first need to obtain the group's ID. To my knowledge, Facebook unfortunately 
# doesn't offer any easy way to do that. 'View Page Source' is one option, but I've found a couple of third-party 
# services like http://wallflux.com/facebook_id/ is much easier to use**   
# 
# The example below uses the **Berkeley CS Group** https://www.facebook.com/groups/berkeleycs/

# <codecell>

SEARCH_LIMIT = 500  # facebook allows 500 max
cal_cs_id = '266736903421190'
cal_cs_feed = g.get_connections(cal_cs_id, 'feed', limit=SEARCH_LIMIT)['data']
pp(cal_cs_feed[0:5])

# <codecell>

cal_cs_feed[0]

# <codecell>

def print_feed(feed):
    '''
    Prints out every post, along with its comments, in a feed
    '''
    for post in feed:
        if 'message' in post:
            msg = strip(to_ascii(post['message']))
            print 'POST:', msg, '\n'
        
        print 'COMMENTS:'
        if 'comments' in post:
            for comment in post['comments']['data']:
                if 'message' in comment:
                    comment = strip(to_ascii(comment['message']))
                    if comment is not None and comment != '':
                        print '+', comment
        print '-----------------------------------------\n'
        
print_feed(cal_cs_feed[:3])

# <codecell>

def find_link(post):
    '''
    Finds the permanent link to a given post
    '''
    if 'actions' in post:
        actions = post['actions']
        for action in actions:
            if 'link' in action:
                return action['link']
    return ''
    
def save_feed(feed):
    '''
    Saves the input feed in a Python list for later processing
    Also strips whitespace and lemmatizes along the way
    '''
    posts = []
    for post in feed:
        if 'message' in post and 'actions' in post:
            msg = strip(to_ascii(post['message']))
            link = strip(to_ascii(find_link(post)))
            posts.append((msg, link))
        
        if 'comments' in post:
            for comment in post['comments']['data']:
                if 'message' in comment and 'actions' in comment:
                    msg = strip(to_ascii(comment['message']))
                    link = strip(to_ascii(find_link(comment)))
                    if msg is not None and msg != '':
                        posts.append((msg, link))
    return posts
                
feed = save_feed(cal_cs_feed)
feed[:3]

# <codecell>

def bag_of_words_tfidf(lst):
    '''
    Constructs a bag of words model, where each document is a Facebook post/comment
    Also applies TFIDF weighting, lemmatization, and filter out stopwords
    '''
    model = Model(documents=[], weight=TFIDF)
    for msg, link in lst:
        doc = Document(msg, stemmer=LEMMA, stopwords=True, name=msg, description=link)
        model.append(doc)
    return model

def cosine_similarity(model, term, num=10):
    '''
    Finds the cosine similarity between the input document and each document in 
    the corpus, and outputs the best 'num' results
    '''
    doc = Document(term, stemmer=LEMMA, stopwords=True, name=term)
    return model.neighbors(doc, top=num)

def process_similarity(result):
    '''
    Processes the result in a nicely formatted table
    
    result is a tuple of length 2, where the first item is the similarity score, 
    and the second item is the document itself
    '''
    from prettytable import PrettyTable
    
    pt = PrettyTable(field_names=['Post', 'Sim', 'Link'])
    pt.align['Post'], pt.align['Sim'], pt.align['Link'] = 'l', 'l', 'l'
    [ pt.add_row([res[1].name[:45] + '...', "{0:.2f}".format(res[0]), 
                  res[1].description]) for res in result ]
    return pt

# <codecell>

# Constructs the bag of words model.
# We don't need to call this function more than once, unless the corpus changed
bag_of_words = bag_of_words_tfidf(feed)

# <markdowncell>

# # Enter your query below, along with the number of results you want to search for

# <codecell>

QUERY = 'declaring major early'
NUM_SEARCH = 10

# <codecell>

sim = cosine_similarity(bag_of_words, QUERY, NUM_SEARCH)
print process_similarity(sim)

# <markdowncell>

# ## compares with

# <markdowncell>

# ![FB Search result](files/img/fb_search_1.png)

# <markdowncell>

# **, which I think is not bad at all. The top result from this search system is:**    
# ![My searcht](files/img/fb_search_2.png)
# 
# , **whereas the top result for FB search is:**    
# ![FB Search](files/img/fb_search_3.png)

# <codecell>


