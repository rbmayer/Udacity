# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:32:11 2015
MongoDB queries on Syria OSM
"""

#Queries
q1 = 'Total number of streets'
q2 = 'Total number of streets with arabic names only'
q3 = 'Total number of streets with english names only'
q4 = 'Total number of streets with both arabic and english names'
q5 = 'Total number of streets with names in other languages'

q1_pipeline = [{ "$match" : { "street_names" : { "$exists" : 1 } } },
               { "$group" : { "_id" : q1, "count" : { "$sum" : 1 } } } ]
q2_pipeline = [{ "$match" : { "street_names.arabic_street_name" : 1, 
                             "street_names.english_street_name" : 
                                 { "$exists" : 0 } } },
               { "$group" : { "_id" : q2, "count" : { "$sum" : 1 } } } ]
q3_pipeline = [{ "$match" : { "street_names.english_street_name" : 1, 
                             "street_names.arabic_street_name" : 
                                 { "$exists" : 0 } } },
               { "$group" : { "_id" : q3, "count" : { "$sum" : 1 } } }  ]
q4_pipeline = [{ "$match" : { "street_names.english_street_name" : 1, 
                             "street_names.arabic_street_name" : 1 } },
               { "$group" : { "_id" : q4, "count" : { "$sum" : 1 } } } ]
q5_pipeline = [{ "$match" : { "street_names" : { "$exists" : 1 }, 
                             "street_names.english_street_name" : 
                                 { "$exists" : 0 }, 
                            "street_names.arabic_street_name" : 
                                { "$exists" : 0 } } },
               { "$group" : { "_id" : q5, "count" : { "$sum" : 1 } } } ]

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
print ('check: ' + str(query1[0]['count']) + ' - ( ' + str(query2[0]['count']) 
        + ' + ' + str(query3[0]['count']) + ' + ' + str(query4[0]['count']) + 
        ' + ' + str(query5[0]['count']) + ' ) = ' + str(query1[0]['count'] - 
        query2[0]['count'] - query3[0]['count'] - query4[0]['count'] - 
        query5[0]['count']))

'''
results:
Total number of streets: 1008
Total number of streets with arabic names only: 756
Total number of streets with english names only: 193
Total number of streets with both arabic and english names: 56
Total number of streets with names in other languages: 3
check: 1008 - ( 756 + 193 + 56 + 3 ) = 0
'''