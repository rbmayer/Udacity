# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:42:59 2015

@author: rebecca

Count number of tags with "addr"
add 'addr' keys and values to dict
output unicode strings
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import codecs

OSMFILE = '/home/rebecca/version-control/nanodegree/project3/map'

def audit(osmfile):
    addr_tag_count = 0
    addr_tags = defaultdict(set)
    addr_text = []
    osm_file = open(osmfile, "r")
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                k, v = tag.attrib['k'], tag.attrib['v']
                if type(k) != unicode:
                    k = unicode(k, encoding='utf-8')                
                if type(v) != unicode:
                    v = unicode(v, encoding='utf-8')
                if u"addr" in k:
                    addr_tags[k].add(v)
                    addr_tag_count = addr_tag_count + 1
                    addr_text.append(v)

    return addr_tag_count, addr_tags, addr_text
    

addr_tag_count, addr_tags, addr_text = audit(OSMFILE)

# results from map (9/15/2015):
# tag_count = 20
# addr:city = 12, addr:street = 6, addr:housenumber = 2


# Total results
'''
def total_results(addr_tags):
    city = 0        
    housenumber = 0
    street = 0
    for k, v in addr_tags.iteritems():
       if 'city' in k:
           city = len(v)
       elif 'street' in k:
           street = len(v)
       elif 'housenumber' in k:
           housenumber = len(v)
    return city, street, housenumber
'''