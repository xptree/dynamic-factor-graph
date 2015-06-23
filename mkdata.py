#!/usr/bin/env python
# encoding: utf-8
# File Name: mkdata.py
# Author: Jiezhong Qiu
# Create Time: 2015/02/09 14:12
# TODO: generate_Y data for dfg

import os
import json
import util
import datetime
import logging
import pickle
from xlrd import open_workbook
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
DEV_PATH = "../../xtx/dev/"
RAW_GRADE_DIR = DEV_PATH + "RawData/grades/"
GRADE_DIR = DEV_PATH + "Data/Excel.json"
FORUM_DIR = DEV_PATH + "Data/Forum.json"
COURSE_INFO_DIR = DEV_PATH + "Data/Course_.json"
BEHAVIOR_DIR = DEV_PATH + "Data/LearningBehavior.json"
LEARNING_TIME_DIR = DEV_PATH + 'Data/Trackinglog.json'
MONGO_DIR = DEV_PATH + 'Data/MongoDB.json'
DEMOGRAPHICS_DIR = DEV_PATH + 'Data/Demographics.json'
EPS = 1e-3

class mkdata(object):
    def __init__(self):
        '''generate_Y data as the following format
            feature[uid][T] is a list of features for user uid at time T
            the feature shoule be additive
            we remove register-only student from the dataset
        '''
        self.feature = {}
        self.feature_num = 0
        with open(COURSE_INFO_DIR) as f:
            courses = json.load(f)
        self.getUser()
        self.start = util.parseDate(courses[self.course]['start'])
        self.end = util.parseDate(courses[self.course]['end'])
        for uid in self.feature:
            for single_date in util.daterange(self.start, self.end):
                self.feature[uid][single_date] = []
        logger.info('course: %s user: %d start: %s end: %s', self.course,
                len(self.feature), self.start.isoformat(), self.end.isoformat())

    def expand_feature(self, num):
        self.feature_num += num
        for uid in self.feature:
            for single_date in self.feature[uid]:
                self.feature[uid][single_date] = [0.] * num + self.feature[uid][single_date]
    def expand_X(self, num):
        self.n_in += num
        for uid in self.X:
            for T in xrange(len(self.ddls) + 1):
                self.X[uid][T] = [0.] * num + self.X[uid][T]
    def getTimeStamp(self, date_obj):
        for index, item in enumerate(self.ddls):
            if date_obj <= item:
                return index
        return len(self.ddls)
    def getUser(self):
        with open(GRADE_DIR) as f:
            grades = json.load(f)[0]
        for uid in grades:
            if self.course in grades[uid] and grades[uid][self.course] > 0:
                self.feature[uid] = {}
    def getLearningData(self):
        # video_time assign_time
        # video_day assign_day
        self.expand_feature(4)
        with open(LEARNING_TIME_DIR) as f:
            learn = json.load(f)
        for uid in learn:
            if uid not in self.feature:
                continue
            if self.course not in learn[uid]:
                continue
            for item in learn[uid][self.course]:
                single_date = util.parseDate(item[0])
                if single_date < self.start or single_date >= self.end:
                    continue
                self.feature[uid][single_date][0] += item[1]
                self.feature[uid][single_date][1] += item[2]
                self.feature[uid][single_date][2] += 1
                self.feature[uid][single_date][3] += 1

    def getBehaviorData(self):
        # video problem
        # in time visit
        # chapter
        #self.expand_feature(3)
        # ddl hit
        self.expand_feature(6)
        with open(BEHAVIOR_DIR) as f:
            behavior = json.load(f)
        with open(MONGO_DIR) as f:
            mongo = json.load(f)
        for uid in behavior:
            if uid not in self.feature:
                continue
            for date in behavior[uid]:
                single_date = util.parseDate(date)
                if single_date < self.start or single_date >= self.end:
                    continue
                for log in behavior[uid][date]:
                    course, catagory = util.parseLog(log)
                    if course == self.course:
                        if log in mongo and mongo[log]['due'] is not None:
                            T_ddl = self.getTimeStamp(util.parseDate(mongo[log]['due']))
                            T =  self.getTimeStamp(single_date)
                            if T_ddl == T:
                                self.feature[uid][single_date][5] += 1
                        if catagory == 'video':
                            self.feature[uid][single_date][0] += 1
                        elif catagory == 'problem':
                            self.feature[uid][single_date][1] += 1
                        elif catagory == 'sequential':
                            self.feature[uid][single_date][2] += 1
                            try:
                                date_obj = mongo[log]['start']
                            except:
                                print log
                                continue
                            if date_obj is None:
                                continue
                            date_obj = util.parseDate(date_obj)
                            if self.getTimeStamp(date_obj) == self.getTimeStamp(single_date):
                                self.feature[uid][single_date][3] += 1
                        elif catagory == 'chapter':
                            self.feature[uid][single_date][4] += 1
    def save(self, fpath='.', fname=None):
        """save a json or pickle representation of data set"""
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.json' or fpathext == '.pkl':
            fpath, fname = os.path.split(fpath)
        elif fname is None:
            # generate_Y filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)
        fabspath = os.path.join(fpath, fname)
        logger.info('Saving to %s ...' % fabspath)
        with open(fabspath, 'wb') as file:
            if fpathext == '.json':
                json.dump(self.feature, file,
                            indent=4, separators=(',', ';'))
            else:
                pickle.dump(self.feature, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dataset(self, fpath='.', fname=None):
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            fpath, fname = os.path.split(fpath)
        elif fname is None:
            # generate_Y filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)
        fabspath = os.path.join(fpath, fname)
        logger.info('Saving dataset to %s shape=(%d, %d, %d)...' % (fabspath, len(self.ddls)+1, len(self.feature), self.feature_num))
        # n_step x n_seq x n_obsv
        n_step = len(self.ddls) + 1
        n_seq = len(self.feature)
        dataset = np.zeros(shape=(n_step, n_seq, self.feature_num))
        for index, uid in enumerate(self.feature):
            assert len(self.feature[uid]) == len(self.ddls) + 1
            for T in xrange(len(self.feature[uid])):
                assert len(self.feature[uid][T]) == self.feature_num
                for i in xrange(self.feature_num):
                    dataset[T][index][i] = self.feature[uid][T][i]
        X = np.zeros(shape=(n_step, n_seq, self.n_in))
        for index, uid in enumerate(self.feature):
            for T in xrange(len(self.X[uid])):
                if len(self.X[uid][T]) != self.n_in:
                    print len(self.X[uid][T]), self.n_in
                assert len(self.X[uid][T]) == self.n_in
                for i in xrange(self.n_in):
                    X[T][index][i] = self.X[uid][T][i]

        with open(fabspath, 'wb') as file:
            pickle.dump((dataset, X), file, protocol=pickle.HIGHEST_PROTOCOL)
        self.dataset = dataset
    def getDDL(self):
        self.ddls = []
        with open(MONGO_DIR) as f:
            mongo = json.load(f)
        for item in mongo:
            try:
                course, categort = util.parseLog(item)
            except:
                continue
            if course == self.course:
                if mongo[item]['due'] is not None:
                    #print item, mongo[item]['due']
                    self.ddls.append(util.parseDate(mongo[item]['due']))
        self.ddls.sort()
        if self.course == "TsinghuaX/20220332_2X/_":
            self.ddls = self.ddls[:-1]
        for item in self.ddls:
            print item, (item - self.start).days / float((self.end - self.start).days)
    def getStageFeature(self):
        feature = {}
        for uid in self.feature:
            feature[uid] = {}
            for single_date in self.feature[uid]:
                #date_str = single_date.isoformat()
                delta = (single_date - self.start).days
                feature[uid][delta] = self.feature[uid][single_date]
        sample = self.ddls + [self.end - datetime.timedelta(1)]
        sample = [(item - self.start).days for item in sample]
        self.feature = {}
        for uid in feature:
             self.feature[uid] = []
             p = 0
             tmp = [0.] * self.feature_num
             for T in xrange(0, (self.end-self.start).days):
                if T <= sample[p]:
                    for i in xrange(self.feature_num):
                        tmp[i] += feature[uid][T][i]
                if T == sample[p]:
                    self.feature[uid].append(tmp)
                    p += 1
                    tmp = [0.] * self.feature_num
    def filte(self, filter_type='binary', threshold=0.3):
        # first merge self.feature and self.score
        self.feature_num += 1
        for uid in self.score:
           for j in xrange(len(self.ddls) + 1):
               self.feature[uid][j].append(self.score[uid][j])
        for i in xrange(self.feature_num):
            for T in xrange(len(self.ddls) + 1):
                tmp = sorted([self.feature[uid][T][i] for uid in self.feature], reverse=True)
                if tmp[0] == 0:
                    tmp[0] = EPS
                if filter_type == 'binary':
                    door = tmp[int(len(self.feature) * threshold)]
                    if door == tmp[0]:
                        door -= EPS
                    elif door == tmp[-1]:
                        door += EPS
                    for uid in self.feature:
                        self.feature[uid][T][i] = 1 if self.feature[uid][T][i] > door else 0
                elif filter_type == 'real':
                    for uid in self.feature:
                        self.feature[uid][T][i] = self.feature[uid][T][i] / float(tmp[0])

    def getDemographics(self):
        # binary feature
        # male, female, el, jhs, hs, c, b, m, p, [0,18], [18,23], [23, 28], [28, 36], [36, 51], [> 51]
        with open(DEMOGRAPHICS_DIR) as f:
            demos = json.load(f)
        self.n_in += 15
        for uid in self.feature:
            tmp = []
            demo = demos[uid]
            for task in ['m', 'f']:
                tmp.append(1 if demo['gender'] == task else 0)
            for task in ['el', 'jhs', 'hs', 'c', 'b', 'm', 'p']:
                tmp.append(1 if demo['education'] == task else 0)
            if demo['age'] is not None:
                age = 2014 - demo['age']
                task = [0, 18, 23, 28, 36, 51, 1000]
                for i in xrange(len(task)-1):
                    tmp.append(1 if age >= task[i] and age < task[i+1] else 0)
            else:
                tmp += [0.] * 6
            for T in xrange(len(self.ddls)+1):
                self.X[uid][T] += tmp
    def getForumData(self):
        # post, reply, replyed, length, upvoted, cert-friend
        self.expand_feature(6)
        with open(FORUM_DIR) as f:
            forum = json.load(f)
        for oid, item in forum.iteritems():
            if item['course'] != self.course:
                continue
            single_date = util.parseDate(item['date'])
            uid = item['user']
            if uid in self.feature and single_date >= self.start and single_date < self.end:
                if item['father'] == None:
                    self.feature[uid][single_date][0] += 1
                else:
                    self.feature[uid][single_date][1] += 1
                    fid = forum[item['father']]['user']
                    if fid in self.feature:
                        self.feature[fid][single_date][2] += 1
                        T = self.getTimeStamp(single_date)
                        if T > 0 and self.score[fid][T-1] > .5:
                            self.feature[uid][single_date][5] += 1
                        if T > 0 and self.score[uid][T-1] > .5:
                            self.feature[fid][single_date][5] += 1
                self.feature[uid][single_date][3] += item['length']
                self.feature[uid][single_date][4] += item['vote_up']
    def getSequentialRelease(self):
        with open(MONGO_DIR) as f:
            mongo = json.load(f)
        self.expand_X(1)
        for item in mongo:
            try:
                course, categort = util.parseLog(item)
            except:
                continue
            if course == self.course:
                if mongo[item]['start'] is not None and item.find('sequential') != -1:
                    print item, mongo[item]['start']
                    date_obj = util.parseDate(mongo[item]['start'])
                    T = self.getTimeStamp(date_obj)
                    for uid in self.X:
                        self.X[uid][T][0] += 1

    def generate_Y(self):
        raise NotImplementedError

    def regenerate(self):
        for uid in self.feature:
            for T in xrange(len(self.ddls) + 1):
                self.X[uid][T] += self.feature[uid][T][:-1]
                self.feature[uid][T] = self.feature[uid][T][-1:]
        self.n_in = self.n_in + self.feature_num - 1
        self.feature_num = 1

    def generate_X(self):
        self.n_in = 0
        self.X = {}
        for uid in self.feature:
            self.X[uid] = [[] for i in xrange(len(self.ddls)+1)]
        # Demographics Feature
        self.getDemographics()
        #self.getSequentialRelease()
        # Course Release Feature
    def base_line(self):
        for i in xrange(5, len(self.ddls) + 1):
            median_tp1 = np.percentile(self.dataset[4,:,-1], 70.3)
            median_t = np.percentile(self.dataset[i,:,-1], 70.3)
            if median_tp1 == 1.:
                median_tp1 -= EPS
            if median_t == 1.:
                median_t -= EPS
            print roc_auc_score(self.dataset[i,:,-1] > median_t, self.dataset[4,:,-1])
            print precision_recall_fscore_support(self.dataset[4,:,-1] > median_tp1, self.dataset[i,:,-1] > median_t, average='micro')
    def dumpScoreDotDat(self, fileName):
        with open(fileName, 'wb') as f:
            title = '\t'.join(['userid'] + ['ps_%d' % i for i in xrange(len(self.score.values()[0]))])
            print >> f, title
            for k, v in self.score.iteritems():
                content = [k] + [str(item) for item in v]
                print >> f, '\t'.join(content)
    def __get_score__(self, scoreColumn, fname):
        book = open_workbook(RAW_GRADE_DIR + fname)
        sheet = book.sheet_by_index(0)
        scores = [sheet.col_values(util.getExcelColumnId(columnStr))
                    for columnStr in scoreColumn]
        self.score = {}
        users = sheet.col_values(0)
        for i in xrange(1, len(users)):
            user = str(int(users[i]))
            if user not in self.feature:
                logger.info('excel break from user %s' % user)
                break
            self.score[user] = []
            for j in xrange(len(scoreColumn)):
                this_score = float(scores[j][i])
                #last_score = 0 if j == 0 or j == len(scoreColumn) - 1 else float(self.score[user][-1])
                last_score = 0
                self.score[user].append(this_score + last_score)

class Circuit(mkdata):
    def __init__(self):
        self.course = "TsinghuaX/20220332_2X/_"
        mkdata.__init__(self)

    def getScore(self):
        # Column D to M
        scoreColumn = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'BJ']
        fname = 'grades_TsinghuaX-20220332_2X-_.xlsx'
        self.__get_score__(scoreColumn, fname)

    def generate_Y(self):
        self.getDDL()
        self.getScore()
        self.getForumData()
        self.getLearningData()
        self.getBehaviorData()
        self.getStageFeature()
        self.filte(filter_type='real', threshold=41. / 411)

class DataStructure(mkdata):
    def __init__(self):
        self.course = "TsinghuaX/30240184_1X/_"
        mkdata.__init__(self)
    def getUser(self):
        with open(GRADE_DIR) as f:
            grades = json.load(f)[0]
        for uid in grades:
            if self.course in grades[uid]:
                self.feature[uid] = {}
    def getScore(self):
        scoreColumn = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        fname = 'grades_TsinghuaX-30240184_1X-_.xlsx'
        self.__get_score__(scoreColumn, fname)


class Combin(mkdata):
    def __init__(self):
        self.course = "TsinghuaX/60240013X/_"
        mkdata.__init__(self)
    def getUser(self):
        with open(GRADE_DIR) as f:
            grades = json.load(f)[0]
        for uid in grades:
            if self.course in grades[uid]:
                self.feature[uid] = {}
    def getScore(self):
        scoreColumn = [chr(ord('D') + i) for i in xrange(ord('Z')-ord('D')+1)] + ['A' + chr(ord('A') + i) for i in xrange(ord('J')-ord('A')+1)]
        print scoreColumn
        fname = 'grades_TsinghuaX-60240013X-_.xlsx'
        self.__get_score__(scoreColumn, fname)

class Finance2014(mkdata):
    def __init__(self):
        self.course = "TsinghuaX/80512073_2014_1X/_2014_"
        mkdata.__init__(self)

    def getScore(self):
        # Column D to M
        scoreColumn = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O']
        fname = 'grades_TsinghuaX-80512073_2014_1X-_.xlsx'
        self.__get_score__(scoreColumn, fname)

    def generate_Y(self):
        self.getDDL()
        self.getScore()
        self.getForumData()
        self.getLearningData()
        self.getBehaviorData()
        self.getStageFeature()
        self.filte(filter_type='real', threshold=0.296)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    '''
    fin2 = Finance2014()
    fin2.generate_Y()
    fin2.generate_X()
    fin2.regenerate()
    fin2.save_dataset('data/fin2.pkl')
    fin2.base_line()
    '''
#    combin = Combin()
#    combin.generate_Y()
#    combin.save('combin.pkl')
    '''
    circuit = Circuit()
    circuit.generate_Y()
    circuit.generate_X()
    circuit.regenerate()
    circuit.save_dataset('data/circuit.pkl')
    '''
    '''
    dsa = DataStructure()
    dsa.getScore()
    dsa.dumpScoreDotDat('data_structure.dat')
    '''
    com = Combin()
    com.getScore()
    com.dumpScoreDotDat('combinatorics.dat')
