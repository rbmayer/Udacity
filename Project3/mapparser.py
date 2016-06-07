#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fill out the count_tags function. It should return a dictionary with the 
tag name as the key and number of times this tag can be encountered in 
the map as value.

"""
import xml.etree.cElementTree as ET

def count_tags(filename):
        # YOUR CODE HERE
    tag_count = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag not in tag_count:
            tag_count[elem.tag] = 1
        else:
            tag_count[elem.tag] = tag_count[elem.tag] + 1
        elem.clear()
        
    return tag_count

OSMFILE = '/home/rebecca/version-control/nanodegree/project3/map'
tag_count = count_tags(OSMFILE)


'''
results:
{'bounds': 1,
 'member': 3616,
 'meta': 1,
 'nd': 652701,
 'node': 540467,
 'note': 1,
 'osm': 1,
 'relation': 227,
 'tag': 179691,
 'way': 70759
'''