import os
import nltk
from nltk.corpus import brown
from cPickle import dump, load


def build_tagger():
    '''
    Backoff tagger starting with bigram tagging. Default tag is NN
    '''
    brown_tagged_sents = brown.tagged_sents()
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
    t2 = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
    with open('./pickle/tagger.pickle', 'wb') as f:
        dump(t2, f, -1)
    return t2

def tagger():
    '''
    Loads the tagger if the pickle file exists, otherwise builds one
    '''
    if os.path.exists('./pickle/tagger.pickle'):
        with open('./pickle/tagger.pickle', 'rb') as f:
            tagger = load(f)
            print 'Successfully loaded POS tagger'
    else:
        tagger = build_tagger()
        print 'Successfully built POS tagger'
    return tagger
