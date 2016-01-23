import sys
from datetime import datetime


class StdOutWithTime(object):

    start = datetime.now()
    file = open('log_%04d_%02d_%02d_%02d_%02d_%02d.txt'
                % (start.year, start.month, start.day, start.hour, start.minute, start.second), 'w')

    def __init__(self, old):
        self.old = old
        self.new_line = True

    def write(self, x):
        if x == '\n':
            self.write2(x)
            self.new_line = True
        elif self.new_line:
            self.write2('%s: %s' % (str(datetime.now()), x))
            self.new_line = False
        else:
            self.write2(x)

    def write2(self, text):
        self.old.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.old.flush()
        self.file.flush()

