# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:32:11 2015

@author: rebecca

MongoDB queries on Syria OSM
"""

#Queries
q1 = 'Total number of documents'
q2 = 'Total number of ways'
q3 = 'Number of documents timestamped since March 15, 2011'
q4 = 'Number of documents timestamped since May 13, 2015'
q5 = 'Number of nodes within 5km of Palmyra timestamped since May 13, 2015: ' # the date ISIS took over Palmyra


q1_pipeline = [{ "$group" : { "_id" : q1, "count" : { "$sum" : 1 } } }  ]
q2_pipeline = [{ "$match" : { "type" : "way" } },
               { "$group" : { "_id" : q2, "count" : { "$sum" : 1 } } } ]
q3_pipeline = [{ "$match" : { "created.timestamp" : { "$gt" : "2011-03-11"} } },
               { "$group" : { "_id" : q3, "count" : { "$sum" : 1 } } } ]
q4_pipeline = [{ "$match" : { "created.timestamp" : { "$gt" : "2015-05-13"} } },
               { "$group" : { "_id" : q4, "count" : { "$sum" : 1 } } } ]
# query nodes within 5km using $geoNear in aggregation pipeline
# note that "maxDistance" does not take the $ operator with $geoNear
# default limit of 100 applies if limit not specified 
q5_pipeline = [{ "$geoNear" : { "spherical" : "true" , "near" : { "type" : "Point" , "coordinates" : [38.2809746, 34.5560155] }, "distanceField" : "dist.calculated", "maxDistance" : 5000, "limit" : 10000 } },
                { "$match" : { "created.timestamp" : { "$gt" : "2015-05-13"} } },
               { "$group" : { "_id" : q4, "count" : { "$sum" : 1 } } }]

               
def get_db(db_name):
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client[db_name]
    return db

def aggregate(db, pipeline):
    result = db.osm_syria.aggregate(pipeline)
    return result

def run_query(db, query, pipeline):    
    result = aggregate(get_db(db), pipeline)
    result = list(result)
    print query + ': ' + str(result[0]['count'])
    return result
    
query1 = run_query('project3', q1, q1_pipeline)
query2 = run_query('project3', q2, q2_pipeline)
query3 = run_query('project3', q3, q3_pipeline)
query4 = run_query('project3', q4, q4_pipeline)
query5 = run_query('project3', q5, q5_pipeline)

# near query using pymongo
from pymongo import MongoClient
client = MongoClient('localhost:27017')
db = client['project3']

# query5: 'Return all nodes within 5 km of Palmyra'
query5_result = []
# when accessing the geospatial index find() must be called on the collection key on which the geospatial index was built, in this case "geometry"
# find() returns a cursor over the set of results
for doc in db.osm_syria.find({"geometry" : { "$near" : { "$geometry" : { "type" : "Point", "coordinates" : [38.2809746, 34.5560155]}, "$maxDistance" : 5000 }}}):
    query5_result.append(doc)

print 'Number of nodes within 5km of Palmyra: ' + str(len(query5_result))




'''
results:


'''



