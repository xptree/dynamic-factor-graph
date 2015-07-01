#!/usr/bin/env python
# encoding: utf-8
# File Name: Config.py
# Author: Jiezhong Qiu

import json
import util
import logging

logger = logging.getLogger(__name__)

#def getDataDir():
#    return '/home/jiezhong/prediction/certificate'
#
#def getStart(course):
#    title = {}
#    title["TsinghuaX/00690242_2015X/2015_T1"] = date(2015, 3, 2)
#    title["TsinghuaX/30240184_2015X/2015_T1"] = date(2015, 3, 3)
#    return title[course]
#
#def getEnd(course):
#    title = {}
#    title["TsinghuaX/00690242_2015X/2015_T1"] = date(2015, 7, 5)
#    title["TsinghuaX/30240184_2015X/2015_T1"] = date(2015, 7, 5)
#    return title[course]
#
#def getDDL(course):
#    if course == "TsinghuaX/30240184_2015X/2015_T1":
#        ddl = []
#        with open('../element.json', 'rb') as f:
#            element = json.load(f)
#        for k,v in element.iteritems():
#            if k.find('30240184') > -1 and v['due'] is not None:
#                dt = datetime.strptime(v['due'], '%Y-%m-%dT%H:%M:%S')
#                dt = util.roundTime(dt, 60 * 60)
#                ddl.append(dt.date())
#        ddl.sort()
#        ddl[-1] = date(2015, 6, 30)
#        return ddl
#    else:
#        raise NotImplementedError
#
#def getThreshold(course):
#    return 0.8
#
#def getPklDir():
#    return '/home/jiezhong/prediction/certificate/dynamic-factor-graph/data/data.pkl'
#
#def getPredictionResultDir():
#    return 'a.txt'

class Config(object):
    def __init__(self, course, fn):
        self.course = course
        with open(fn, 'rb') as f:
            self.config = json.load(f)
        logger.info('Loading config for %s from file %s', course, fn)

    def getThreshold(self):
        # return the threshold of grade
        return self.config['threshold']

    def getPklFile(self):
        # return the pkl file which stores the feature vector for each user in every timestamp
        return self.config["pklFile"]

    def getPredictionResultDir(self):
        # return the prediction result dir
        return self.config["predictionResultDir"]

    def getJsonDir(self):
        # returen json dir which store the result of Feature.py
        return self.config["jsonDir"]

    def getStart(self):
        # return the start of the course
        return util.parseDate(self.config['start'])

    def getEnd(self):
        # return the end of the course
        return util.parseDate(self.config['end'])

    def getDDL(self):
        # a list of ddls
        return [util.parseDate(item) for item in self.config['ddl']]

if __name__ == '__main__':
    config = Config("TsinghuaX/30240184_2015X/2015_T1", 'dsa.json')
