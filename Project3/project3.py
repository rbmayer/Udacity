# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:13:02 2015

@author: rebecca

Project 3: Audit OpenStreetMaps data
"""

import xml.etree.cElementTree as ET
# import pprint
import re
import codecs
import json

highway_types = [u'motorway', u'trunk', u'primary', u'secondary', u'tertiary', 
                 u'unclassified', u'residential', u'service', u'living street', 
                 u'pedestrian', u'track', u'bus_guideway', u'road', 
                 u'proposed', u'construction']
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
# previous script showed no problem chars in the dataset
# problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
mapping = { u"‫ِشارع‬‎": u"شارع",
            u"St.": u"Street",
            u"Rd" : u"Road" }

osmfile = '/home/rebecca/version-control/nanodegree/project3/map' 

def update_name(name, mapping):
    # Find mapping key within name and replace with mapping value
    for key in mapping:
        if name in key:
            name = name.replace(key, mapping[key])
    return name
    
    
def is_name(k):
    return (u'name' in k)
    
    
def is_arabic(unicode_string):
    # search for any word within arabic unicode range (0600, 06ff)
    return re.search(ur'\b[\u0600-\u06ff]*\b', unicode_string, 
                     re.UNICODE).group()
        
        
def is_english(unicode_string):
    # search for any word within ascii unicode range (0000,007f)   
    return re.search(ur'\b[\u0000-\u007f]*\b', unicode_string, 
                     re.UNICODE).group()
    
def make_pairs_unicode(k, v):
    if type(k) != unicode:
        k = unicode(k, encoding='utf-8')
    if type(v) != unicode:
        v = unicode(v, encoding='utf-8')
    return k, v
    
def process_top_level(node, element):
    created = {}
    for k, v in element.attrib.iteritems():
        k, v = make_pairs_unicode(k, v)
        if k == u'id' or k == u'visible':
            node[k] = v
        elif k == 'lat':
            lat = float(v)
        elif k == 'lon':
            lon = float(v)
        else: 
            created[k] = v
    if ('lat' in locals()) and ('lon' in locals()):
        node[u'geometry'] = {"type": "Point", "coordinates": [lon, lat]}
    if created != {}:
        node[u'created'] = created


def process_node(node, element):
    node[u'type'] = u'node'
    # process and clean street addresses
    address = {}
    for tag in element.iter("tag"):
        k, v = make_pairs_unicode(tag.attrib['k'], tag.attrib['v'])
        # ignore 'addr' tags with 2 or more colons
        if k[:4] == u'addr':
            if k.count(':') == 1:
                # clean street names
                if u'street' in k:
                    v = update_name(v, mapping)
                address[k[5:]] = v
        else:
            node[k] = v
    if address != {}:
        node[u'address'] = address

        
def process_way(node, element):
    node[u'type'] = u'way'
    # process nd refs
    for ndtag in element.iter("nd"):
        if u'node_refs' not in node:
            node[u'node_refs'] = [ndtag.attrib['ref']]
        else:
            node[u'node_refs'].append(ndtag.attrib['ref'])
    # process streets
    street_names = {}
    is_street = 0
    # check if way is a street
    for tag in element.iter("tag"):
        k, v = make_pairs_unicode(tag.attrib['k'], tag.attrib['v'])
        if (k == u"highway") and (v in highway_types):
            is_street = 1
    # process all tags
    for tag in element.iter("tag"):
        k, v = make_pairs_unicode(tag.attrib['k'], tag.attrib['v'])
        # clean and process streets
        if is_street == 1:
            if is_name(k):
                v = update_name(v, mapping)
                street_names[k] = v
                # add language fields
                if is_english(v):
                    street_names[u'english_street_name'] = 1
                if is_arabic(v): 
                    street_names[u'arabic_street_name'] = 1
            else:
                node[k] = v
        # process non-street tags
        else:
            node[k] = v
    if street_names != {}:
        node[u'street_names'] = street_names
                             
                            
def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        process_top_level(node, element)
        if element.tag == "node":
            process_node(node, element) 
        if element.tag == "way":
            process_way(node, element)
        return node
    else:
        return None


def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    with codecs.open(file_out, "w") as fo:
        # root-clearing code from http://effbot.org/zone/element-iterparse.htm
        context = ET.iterparse(file_in, events=("start", ))
        context = iter(context)
        event, root = context.next()
        for event, elem in context:
            node = shape_element(elem)
            elem.clear()
            root.clear()
            if node:
                # check unicode-related output code
                if pretty:
                    fo.write(json.dumps(node, indent=2, 
                            ensure_ascii=False).encode('utf8')+"\n")
                else:
                    fo.write(json.dumps(node,  
                            ensure_ascii=False).encode('utf8') + "\n")   
    print "output to JSON completed"
    return 
    
process_map(osmfile)

''' 
Output data format:
Nodes
{    "amenity": "place_of_worship", 
     "name:en": "Khusruwiyah Mosque", 
     "name": "جامع الخسروية", 
     "created": {
                 "changeset": "2951283", 
                 "user": "Esperanza36", 
                 "version": "2", 
                 "uid": "83557", 
                 "timestamp": "2009-10-25T21:36:44Z"
                }, 
    "geometry": {"type": "Point", "coordinates": [37.1606256, 36.1968424]}, 
    "religion": "muslim", 
    "type": "node", 
    "id": "338917163"
}
Ways
{   "node_refs": ["540035110", "2134518476", "2134518475", "540035112", "540035114", "1836302197", "1839880165", "540035119", "1836302196", "1832048528", "540035224"], 
    "created": {
                 "changeset": "17735363", 
                 "user": "Esperanza36", 
                 "version": "4", 
                 "uid": "83557", 
                 "timestamp": "2013-09-08T15:49:23Z"
                }, 
    "street_names": {
                 "name": "نزلة العقبة", 
                 "arabic_street_name": 1
                 }, 
    "type": "way", 
    "id": "173161427", 
    "highway": "residential", 
    "history": "Retrieved from v9"
}
'''
