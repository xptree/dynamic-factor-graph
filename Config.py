#!/usr/bin/env python
# encoding: utf-8
# File Name: Config.py
# Author: Jiezhong Qiu

import json
from datetime import datetime, date
import util

def getDataDir():
    return '/home/jiezhong/prediction/certificate'

def getStart(course):
    title = {}
    title["TsinghuaX/00690242_2015X/2015_T1"] = date(2015, 3, 2)
    title["TsinghuaX/30240184_2015X/2015_T1"] = date(2015, 3, 3)
    return title[course]

def getEnd(course):
    title = {}
    title["TsinghuaX/00690242_2015X/2015_T1"] = date(2015, 6, 24)
    title["TsinghuaX/30240184_2015X/2015_T1"] = date(2015, 6, 24)
    return title[course]

def getDDL(course):  
    if course == "TsinghuaX/30240184_2015X/2015_T1":
        ddl = []
        with open('../element.json', 'rb') as f:
            element = json.load(f)
        for k,v in element.iteritems():
            if k.find('30240184') > -1 and v['due'] is not None:
                dt = datetime.strptime(v['due'], '%Y-%m-%dT%H:%M:%S')
                dt = util.roundTime(dt)
                ddl.append(dt.date())
        ddl.sort()
        return ddl
    else:
        raise NotImplementedError

def getThreshold(course):
    return 0.7

if __name__ == '__main__':
    print getDDL("TsinghuaX/30240184_2015X/2015_T1")
