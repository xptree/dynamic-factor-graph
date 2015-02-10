#!/usr/bin/env python
# encoding: utf-8
# File Name: util.py
# Author: Jiezhong Qiu
# Create Time: 2015/02/09 15:11
# TODO:

from datetime import datetime
from datetime import timedelta
import re

pattern = re.compile(r"i4x://(?P<org>[^/]*)/(?P<course>[^/]*)/(?P<catagory>[^/]*)/(?P<oid>\w{32})")

def parseDate(dateStr):
    return datetime.strptime(dateStr.split("T")[0], "%Y-%m-%d").date()

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def parseLog(logStr):
    for m in pattern.finditer(logStr):
        content = m.groupdict()
        course = "%s/%s/_" % (content["org"], content["course"])
        if course == 'TsinghuaX/80512073_2014_1X/_':
            course += '2014_'
        return course, content['catagory']

def getExcelColumnId(columnStr):
    tmp, column = columnStr, 0
    for ch in tmp:
        column = column * 26 + ord(ch) - ord('A')
    column += sum([26**i for i in xrange(len(tmp))])
    return column - 1
if __name__ == "__main__":
    pass


