#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
"""
check the "k" value for each "<tag>" and see if they can be valid keys in MongoDB,
as well as see if there are any other potential problems.

We have provided you with 3 regular expressions to check for certain patterns
in the tags. As we saw in the quiz earlier, we would like to change the data model
and expand the "addr:street" type of keys to a dictionary like this:
{"address": {"street": "Some value"}}
So, we have to see if we have such tags, and if we have any tags with problematic characters.
Please complete the function 'key_type'.
"""


lower = re.compile(r'^([a-z]|_)*$') # alphanumeric only
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$') # alphanumeric strings separated by colon
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


def key_type(elem, keys):
    if elem.tag == "tag":
        # ElementTree .attrib function return list of attributes in a dictionary
        for key, value in elem.attrib.iteritems():
            if key == 'k':
                # use re.match to find pattern at beginning of string
                if lower.match(value) != None:
                    keys['lower'] = keys['lower'] + 1
                elif lower_colon.match(value) != None:
                    keys['lower_colon'] = keys['lower_colon'] + 1
                # use re.search to find problem chars anywhere in the string
                elif problemchars.search(value) != None:
                    keys['problemchars'] = keys['problemchars'] + 1
                else:
                    keys['other'] = keys['other'] + 1
                    
    return keys



def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys



filein = '/home/rebecca/version-control/nanodegree/project3/map'
keys = process_map(filein)
pprint.pprint(keys)

'''
results: {'lower': 153584, 'lower_colon': 21432, 'other': 4675, 'problemchars': 0}
no problem chars
'''