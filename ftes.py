# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Connects to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBANbufuqn7hIhZCtTnmdeLlhLRaNz02JbCoA5f3i9TxpaCjBRktyqgkQwpSuO7nTJC3UXtj3YDrh7WFLp0ELEZAM6rgV1mQFp8ytzdsiTk4ZB0LE9NwohpPu9hgBvjfkR5h7He13GZBSzA92LtRcKplMx56NQP7uLu7evbsw7EIUZACi9OnRDtSOQNlzIjzQZDZD'

# <markdowncell>

# # Import packages and dependencies

# <codecell>

import facebook  # pip install facebook-sdk, not facebook

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
    
def stripPunctuation(s):
    '''
    Strips punctuation from a string
    '''
    return s.translate(table, string.punctuation)
    
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

# ## Seriously, Facebook?
# 
# ![Facebook search is so easy to see!](files/img/fb_search_1.png)

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
pp(cal_cs_feed)

# <codecell>

len(cal_cs_feed)

# <codecell>

def print_feed(feed):
    '''
    Prints out every post, along with its comments, in a feed
    '''
    for post in feed:
        msg = post['message'].encode('ascii', 'ignore').\
            replace('\n', ' ').replace('\r', '').strip()
        print 'POST:', msg, '\n'
        print 'COMMENTS:'
        
        if 'comments' in post:
            for comment in post['comments']['data']:
                comment = comment['message'].encode('ascii', 'ignore').\
                    replace('\n', ' ').replace('\r', '').strip()
                if comment is not None and comment != '':
                    print '+', comment
        print '-----------------------------------------\n'
        
print_feed(cal_cs_feed[:5])

# <codecell>

def save_feed(feed):
    '''
    Saves the feed as a Python cPickle file for later processing
    '''
    posts = []
    for post in feed:
        msg = post['message']
        posts.append(msg)
        
        if 'comments' in post:
            for comment in post['comments']['data']:
                comment = comment['message'].encode('ascii', 'ignore').\
                    replace('\n', ' ').replace('\r', '').strip()
                posts.append(comment)

