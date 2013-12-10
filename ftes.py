# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Connect to Facebook    
# Login to your Facebook account and go to https://developers.facebook.com/tools/explorer/ to obtain and set permissions for an access token.

# <codecell>

ACCESS_TOKEN = 'CAACEdEose0cBAGmngXljzIoIg84zDWFeuxGxO1ixulIDUkZBhzqMljCvhJnBwwR2gSKuG1lYGfsfoCicJ4MCNcCjXmIbYZC1pAd2Opvcowum88DImGSBxbUXOPhop2IPKSbjgLvAxc72znespZBgvMCzB4KZBrynvo3noo2HnMc9CLnaXuZAFDGDODoNYz3KSlOZCQZC7SdIQZDZD'

# <markdowncell>

# # Import packages and dependencies

# <codecell>

import requests
import json
import facebook  # pip install facebook-sdk, not facebook
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
%matplotlib inline

from prettytable import PrettyTable
from collections import Counter

# <codecell>

def pp(o):
    '''
    A helper function to pretty-print Python objects as JSON
    '''
    print json.dumps(o, indent=1)

# <codecell>

# Creates a connection to the Graph API with your access token
g = facebook.GraphAPI(ACCESS_TOKEN)

# <codecell>

g.get_connections("me", "friends")

