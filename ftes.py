# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Khoa Tran - INFO 256 Fall 2013
# # Final Project Demo
# # Facebook Topics Extraction System

# <markdowncell>

# ## Project Goals:
# 
# * Identify popular topics in a given facebook page
# * Find similar posts related to a user's interest, and compare the results with facebook's

# <markdowncell>

# ## Motivation
# 
# * FB Search seems to be missing some results
# * See how well a simple TFIDF model would perform against FB Search
# * Finally... find out some popular topics being discussed without reading thru every post

# <markdowncell>

# ![](files/img/fb_search_1.png)

# <markdowncell>

# # Connects to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBAMkOPptLZCUoAlRsUfD5nOnLcDo32ce6KZBKCxhOJ6E4YBjrQAFAxmoWmIdyd8AWl1TJoghswmnUKrTixUB5Toil9FsyfTy8YLjCfq292flRfYIJML0nzf3QwpR5nOnOGXLfntTlvriHnrXZBRQUChXMv3NPNbPTdeoKdgxDuXt7QCpfhVO8GCUXcNZAggZDZD'
SEARCH_LIMIT = 500  # facebook allows 500 max

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
from prettytable import PrettyTable
from collections import defaultdict, Counter

# NLP!
import string
import nltk
from nltk.corpus import stopwords
import tagger as tag
import enchant
from nltk.metrics import edit_distance
from pattern.vector import Document, Model, TFIDF, LEMMA, KMEANS, HIERARCHICAL, COSINE

# <markdowncell>

# ### Lemmatizer, stemmer, and spelling corrector

# <codecell>

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
table = string.maketrans("", "")
sw = stopwords.words('english')

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

# ### Some tiny examples for error correction

# <codecell>

# Works
print correct('aninal')
print correct('behols'), '\n'

# Doesn't work... very well
print correct('aisplane') # airplane
print spell_dict.suggest('aisplane')

# <markdowncell>

# ## Creates a connection to the Graph API with your access token

# <codecell>

g = facebook.GraphAPI(ACCESS_TOKEN)

# <markdowncell>

# # Part I: Retrieves a group's feed, and builds a simple search engine with TFIDF and Cosine Similarity

# <markdowncell>

# **To retrieve a group's feed, you first need to obtain the group's ID. To my knowledge, Facebook doesn't offer any easy way to do that. 'View Page Source' is one option, but I've found a couple of third-party 
# services like http://wallflux.com/facebook_id/ is much easier to use**   
# 
# The example below uses the **Berkeley CS Group** https://www.facebook.com/groups/berkeleycs/

# <codecell>

# Only needs to make connection once
cal_cs_id = '266736903421190'
cal_cs_feed = g.get_connections(cal_cs_id, 'feed', limit=SEARCH_LIMIT)['data']

# <codecell>

pp(cal_cs_feed[15])

# <codecell>

len(cal_cs_feed)

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
        
print_feed(cal_cs_feed[10:12])

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
feed[30:35]

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

# ![](files/img/fb_search_0.png)

# <markdowncell>

# **, which I think is not bad at all. The top result from this search system is:**    
# ![](files/img/fb_search_2.png)
# 
# , **whereas the top result for FB search is:**    
# ![](files/img/fb_search_3.png)

# <markdowncell>

# # Part II: What are the most popular topics in a group's feed right now?

# <codecell>

sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''

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

def leaves(tree):
    '''
    Finds NP (nounphrase) leaf nodes of a chunk tree
    '''
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()

def normalize(word):
    '''
    Normalizes words to lowercase and stems/lemmatizes it
    '''
    word = word.lower()
    #word = stem(word)
    word = strip(lemmatize(word), True)
    return word

def acceptable_word(word):
    '''
    Checks conditions for acceptable word: valid length and no stopwords
    '''
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in sw)
    return accepted

def get_terms(tree):
    '''
    Gets all the acceptable noun_phrase term from the syntax tree
    '''
    for leaf in leaves(tree):
        term = [normalize(w) for w, t in leaf if acceptable_word(w)]
        yield term

def extract_noun_phrases(text):
    '''
    Extracts all noun_phrases from a given text
    '''
    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = tagger.tag(toks)

    # Builds a POS tree
    tree = chunker.parse(postoks)
    terms = get_terms(tree)

    # Extracts Noun Phrase
    noun_phrases = []
    for term in terms:
        np = ""
        for word in term:
            np += word + " "
        if np != "":
            noun_phrases.append(np.strip())
    return noun_phrases

# <codecell>

def extract_feed(feed):
    '''
    Extracts popular topics (noun phrases) from a feed, and builds a simple
    counter to keep track of the popularity
    '''
    topics = defaultdict(int)
    for post, link in feed:
        noun_phrases = extract_noun_phrases(post)
        for np in noun_phrases:
            topics[np] += 1
    return topics

# <codecell>

topics = extract_feed(feed)
c = Counter(topics)
c.most_common(20)

# <codecell>

from pytagcloud import create_tag_image, create_html_data, make_tags, LAYOUT_HORIZONTAL, LAYOUTS
from pytagcloud.colors import COLOR_SCHEMES
from operator import itemgetter

def get_tag_counts(counter):
    '''
    Get the noun phrase counts for word cloud by first converting the counter to a dict
    '''
    return sorted(dict(counter).iteritems(), key=itemgetter(1), reverse=True)
    
def create_cloud(counter):
    '''
    Creates a word cloud from a counter
    '''
    tags = make_tags(get_tag_counts(counter)[:80], maxsize=120, 
                     colors=COLOR_SCHEMES['goldfish'])
    create_tag_image(tags, './img/cloud_large.png', 
                     size=(900, 600), background=(0, 0, 0, 255), 
                     layout=LAYOUT_HORIZONTAL, fontname='Lobster')

# <codecell>

create_cloud(c)

# <markdowncell>

# ![](files/img/cloud_large.png)

# <markdowncell>

# # Results:
# 
# Part I:
# 
# * Tried out at least 5 different queries with the simple TFIDF search, and the result seems to be on par with FB Search. The ranking, as seen above, is different, however.      
# 
# * Revisit and try out different algorithms, including stemming, typo correction, variants of TFIDF (maxmimum TF normalization, sublinear scaling, etc.) http://nlp.stanford.edu/IR-book/html/htmledition/variant-tf-idf-functions-1.html
# 
# Part II:
# 
# * There are quite some noises ('http', 'www', etc.) and words that don't really tell you anything new ('class', 'course', etc.)    
# 
# * The overall result seems consistent with what most CS students usually discuss on FB ('telebears', '61B', etc.)
# 
# Overall a great project to work on and extend in the future!

# <markdowncell>

# # Thanks!

