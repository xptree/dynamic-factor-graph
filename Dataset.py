#!/usr/bin/env python
# encoding: utf-8
# File Name: mkdata.py
# Author: Jiezhong Qiu

import logging
import Config as config
import os
from bson import json_util
import util

class Dataset(object):
    def __init__(self):
        '''generate_Y data as the following format
            feature[uid][T] is a list of features for user uid at time T
            the feature shoule be additive
            we remove register-only student from the dataset
        '''
        self.feature = {}
        self.feature_num = 0
        self.path = config.getDataDir()
        self.getUser()
        self.start = config.getStart(self.course)
        self.end = config.getEnd(self.course)
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
    def getUser(self):
        with open(os.path.join(self.path, 'user.json'), 'rb') as f:
            user = json.load(f)
        for uid in user:
            self.feature[uid] = {}

    def getForumData(self):
        # post, reply, replyed, length, upvoted
        self.expand_feature(5)
        with open(os.path.join(self.path, 'forum.json')) as f:
            forum = json.load(f, object_hook=json_util.object_hook)
        for oid, item in forum.iteritems():
            single_date = item['date']
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
					if T > 0:
						self.feature[uid][single_date][5] += self.score[fid][T-1]
						self.feature[fid][single_date][5] += self.score[uid][T-1]
                self.feature[uid][single_date][3] += item['length']
                self.feature[uid][single_date][4] += item['vote_up']

    def getScore(self):
        fileDir = os.path.join(self.path, 'grade.json')
        with open(fileDir, 'rb') as f:
            self.score = json.load(f)
    def getLearningData(self):
        # video_time assign_time
		# video_day assign_day
        self.expand_feature(4)
        with open(os.path.join(self.path, 'duration.json')) as f:
            learn = json.load(f)
        for uid in learn:
            if uid not in self.feature:
                continue
            for k, v in learn[uid].iteritems():
                single_date = util.parseDate(k)
                if single_date < self.start or single_date >= self.end:
                    continue
                self.feature[uid][single_date][0] += v[0]
                self.feature[uid][single_date][1] += v[1]
				self.feature[uid][single_date][2] += v[0] > 0
				self.feature[uid][single_date][3] += v[1] > 0

    def getBehaviorData(self):
        # video problem sequential chapter ddl_hit
        self.expand_feature(5)
        with open(os.path.join(self.path, 'behavior.json')) as f:
            behavior = json.load(f)
		with open(os.path.join(self.path, 'element.json')) as f:
            element = json.load(f, object_hook=json_util.object_hook)
        for uid in behavior:
            if uid not in self.feature:
                continue
            for date in behavior[uid]:
                single_date = util.parseDate(date)
                if single_date < self.start or single_date >= self.end:
                    continue
				course, catagory = util.parseLog(log)
                for log in behavior[uid][date]:
					if element[log]['due'] is not None:
						if single_date <= element[log]['due']:
							self.feature[uid][single_date][4] += 1
                    if catagory == 'video':
                        self.feature[uid][single_date][0] += 1
                    elif catagory == 'problem':
                        self.feature[uid][single_date][1] += 1
					elif catagory == 'sequential':
						self.feature[uid][single_date][2] += 1
					elif catagory == 'chapter':
						self.feature[uid][single_date][3] += 1


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
        self.ddls = config.getDDL(self.course)

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
                door = tmp[int(len(self.feature) * threshold)]
                if door == tmp[0]:
                    door -= EPS
                elif door == tmp[-1]:
                    door += EPS
                for uid in self.feature:
                    self.feature[uid][T][i] = 1 if self.feature[uid][T][i] > door else 0
    def getDemographics(self):
        # binary feature
        # male, female, el, jhs, hs, c, b, m, p, [0,18], [18,23], [23, 28], [28, 36], [36, 51], [> 51]
        with open(os.path.join(self.path, 'profile.json')) as f:
            demos = json.load(f)
        self.n_in = 15
        for uid in self.feature:
            tmp = []
            demo = demos[uid]
            for task in ['m', 'f']:
                tmp.append(1 if demo['gender'] == task else 0)
            for task in ['el', 'jhs', 'hs', 'c', 'b', 'm', 'p']:
                tmp.append(1 if demo['education'] == task else 0)
            if demo['age'] is not None:
                age = 2015 - demo['age']
                task = [0, 18, 23, 28, 36, 51, 1000]
                for i in xrange(len(task)-1):
                    tmp.append(1 if age >= task[i] and age < task[i+1] else 0)
            else:
                tmp += [0.] * 6
            for T in xrange(len(self.ddls)+1):
                self.X[uid][T] += tmp

    def generate_Y(self):
        self.getDDL()
        self.getScore()
        self.getForumData() 
        self.getLearningData() 
        self.getBehaviorData() 
        self.getStageFeature()
		threshold = config.getThreshold()
		self.filte(filter_type='binary', threshold=threshold)

    def generate_X(self):
        self.X = {}
        for uid in self.feature:
            self.X[uid] = [[] for i in xrange(len(self.ddls)+1)]
        # Demographics Feature
        self.getDemographics()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = Dataset()
    dataset.generate_Y()
    dataset.generate_X()
    dataset.save_dataset('data.pkl')
