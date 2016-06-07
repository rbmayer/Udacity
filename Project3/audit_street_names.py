# -*- coding: utf-8 -*-
"""
Audit street names

The purpose of this script is to audit english- and arabic-language street names in a segment of the OSM Syria map to identify abbreviations or errors for cleaning at a later stage. Street names generally appear in "way" elements within the "highway" category where they are specified in a "name" tag. However, neither "highway" nor "name" tags refer uniquely to streets. A very small number of street names (6) appear in "node" elements signified by "addr:street". 
"""


import xml.etree.cElementTree as ET
from collections import defaultdict
import re

OSMFILE = '/home/rebecca/version-control/nanodegree/project3/map'

expected = [u"Street", u"Avenue", u"Boulevard", u"Sharia", u"Souq", u"Place", u"Road", u"Roundabout", u"سوق",u"جادة", u"شارع", u"طريق"]

highway_types = [u'motorway', u'trunk', u'primary', u'secondary', u'tertiary', u'unclassified', u'residential', u'service', u'living street', u'pedestrian', u'track', u'bus_guideway', u'road', u'proposed', u'construction']

def match_arabic_name(unicode_name):
    # input unicode string
    # output match for potential street names at beginning of string
    m = re.match(ur'^\S+', unicode_name, re.UNICODE)
    if m:
        street_type = m.group().strip(u"\u202b")
    return street_type
    
def match_english_name(unicode_name):
    # input unicode string
    # output match for potential English-style street names at end of string
    m_end = re.search(ur'\b\S+\.?$', unicode_name, re.IGNORECASE | re.UNICODE)
    # output match for potential transliterated Arabic-style street names at beginning of string
    m_start = re.match(ur'\S+\.?\b', unicode_name, re.IGNORECASE | re.UNICODE)
    # if first word is an arabic-style transliteration return it, otherwise return last word
    if m_start in expected:
        street_type = m_start.group()
    else:
        street_type = m_end.group()
    return street_type
    
def is_arabic(unicode_string):
    # search for any word within arabic unicode range (0600, 06ff)
    return re.search(ur'\b[\u0600-\u06ff]*\b', unicode_string, re.UNICODE).group()
        
def is_english(unicode_string):
    # search for any word within ascii unicode range (0000,007f)   
    return re.search(ur'\b[\u0000-\u007f]*\b', unicode_string, re.UNICODE).group()


def audit_street_type(street_name):
    # convert any bytestrings to unicode
    if type(street_name) != unicode:
        street_name = unicode(street_name, encoding='utf-8')
    # match entries with both english and arabic words as english
    if is_english(street_name): 
        street_type = match_english_name(street_name)
    elif is_arabic(street_name):
        street_type = match_arabic_name(street_name)
    # skip entries in all other languages
    else:
        return None, None
    return street_type, street_name 


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")
    
    
def is_name(elem):
    return ('name' in elem.attrib['k'])
    
    
def add_street_type(street_types, street_type, street_name):
    if street_type not in expected:
        street_types[street_type].add(street_name)    


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        # audit street names in "nodes" containing "addr:street"
        if elem.tag == "node":
            for tag in elem.iter("tag"):
                if is_street_name(tag) and tag.attrib['v'] != "":
                    street_type, street_name = audit_street_type(tag.attrib['v'])
                    add_street_type(street_types, street_type, street_name)
        # audit street names in "ways" containing "highway"
        elif elem.tag == "way":
            highway = 0
            # check if way matches an included highway type
            for tag in elem.iter("tag"):
                if (tag.attrib['k'] == "highway") and (tag.attrib['v'] in highway_types):
                    highway = 1
            if highway == 1:
                for tag in elem.iter("tag"):
                    if is_name(tag) and tag.attrib['v'] != "":
                        street_type, street_name = audit_street_type(tag.attrib['v'])
                        if street_type != None:
                            add_street_type(street_types, street_type, street_name)
                            
    return street_types

street_types = audit(OSMFILE)
street_list = {}
for k in street_types:
   street_list[k] = list(street_types[k])
'''
results: 141 elements returned. add to mapping u"‫ِشارع‬‎" u"St.", u"Rd"

mapping = { u"‫ِشارع‬‎": u"شارع",
            u"St.": u"Street",
            u"Rd" : u"Road"        
            }
'''