#!/usr/bin/env python
# encoding: utf-8
# File Name: Config.py
# Author: Jiezhong Qiu

import json
from datetime import datetime
from bson import json_util

def getDataDir():
    return '/home/jiezhong/prediction/certificate/data'

def getStart(course):
    title = {}
    title["TsinghuaX/00690242_2015X/2015_T1"] = datetime(2015, 3, 2, 2)
    title["TsinghuaX/30240184_2015X/2015_T1"] = datetime(2015, 3, 3)
    return title[course]

def getEnd(course):
    title = {}
    title["TsinghuaX/00690242_2015X/2015_T1"] = datetime(2015, 6, 24, 0)
    title["TsinghuaX/30240184_2015X/2015_T1"] = datetime(2015, 6, 24, 0)
    return title[course]

def getDDL(course):  
    with open('../element.json', 'rb') as f:
        element = json.load(f, object_hook=json_util.object_hook)
    for k,v in element.iteritems():
        if k.find('30240184') > -1 and v['due'] is not None:
            print v['due']

if __name__ == '__main__':
    getDDL("TsinghuaX/00690242_2015X/2015_T1")
