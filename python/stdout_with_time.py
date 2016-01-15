import sys
from datetime import datetime


class StdOutWithTime(object):

    def __init__(self, old):
        self.old = old
        self.new_line = True

    def write(self, x):
        if x == '\n':
            self.old.write(x)
            self.new_line = True
        elif self.new_line:
            self.old.write('%s: %s' % (str(datetime.now()), x))
            self.new_line = False
        else:
            self.old.write(x)

    def flush(self):
        self.old.flush()

