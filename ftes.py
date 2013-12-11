# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Connect to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBANbufuqn7hIhZCtTnmdeLlhLRaNz02JbCoA5f3i9TxpaCjBRktyqgkQwpSuO7nTJC3UXtj3YDrh7WFLp0ELEZAM6rgV1mQFp8ytzdsiTk4ZB0LE9NwohpPu9hgBvjfkR5h7He13GZBSzA92LtRcKplMx56NQP7uLu7evbsw7EIUZACi9OnRDtSOQNlzIjzQZDZD'

# <markdowncell>

# # Import packages and dependencies

# <codecell>

import facebook  # pip install facebook-sdk, not facebook
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
%matplotlib inline

import requests
import json
import simplejson as sj

# <codecell>

def pp(o):
    '''
    A helper function to pretty-print Python objects as JSON
    '''
    print json.dumps(o, indent=2)

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


