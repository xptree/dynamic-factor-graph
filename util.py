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

def roundTime(dt=None, roundTo=60):
    # http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python/10854034#10854034
    """
    Round a datetime object to any time laps in seconds
        dt : datetime.datetime object, default now.
        roundTo : Closest number of seconds to round to, default 1 minute.
        Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None : dt = datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + timedelta(0,rounding-seconds,-dt.microsecond)

def parseDate(dateStr):
    #return dateutil.parser.parse(dateStr).date() 
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


