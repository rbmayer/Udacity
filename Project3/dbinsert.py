# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:39:30 2015

@author: rebecca
"""

import json
import pymongo

''' Use db.osm_syria.remove() before re-running this script to avoid duplicating db's '''

def insert_data(data, db):
    # Insert the data into a collection 'osm_syria'
    db.osm_syria.insert_many(data)


OSMFILE = '/home/rebecca/version-control/nanodegree/project3/map.json'


from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.project3

# http://stackoverflow.com/questions/12451431/loading-and-parsing-a-json-file-in-python
data = []
with open(OSMFILE) as f:
    for line in f:
        data.append(json.loads(line))
    insert_data(data, db)
    
# create 2dsphere geospatial index
db.osm_syria.create_index([("geometry", pymongo.GEOSPHERE)])
# to check index: db.osm_syria.index_information()