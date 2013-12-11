# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Connect to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBAM3TUmgXtVSCCvdHoncQqpKbp6WrCrGuNMQtgBsZBiJtGwZAKZA1bC5CGKhtCNnURflb9L1GVy51rOiAfJvOl1nE302TApqemz2om6ZAjZAlOVNOURIgyZAYYGTq0S94TI1GSay0Jif86ZCKSpiAggbKK3byqBIbCVIvKtfRQrScupeZCSWSTs3NHcH7nD5RbQZDZD'

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
    print json.dumps(o, indent=1)

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

# <codecell>

# Retrieves a group
# The example below uses the Berkeley CS Group
cal_cs_id = '266736903421190'
pp(g.get_connections(cal_cs_id, 'feed'))

