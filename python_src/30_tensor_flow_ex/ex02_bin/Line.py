# encoding: utf-8

import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    pass
pass

class Line:
    def __init__(self, p=None , q=None, line = None ):
        if line is not None :
            self.p = Point(line[0], line[1])
            self.q = Point(line[2], line[3])
        else :
            self.p = p
            self.q = q
        pass
    pass

    def length(self):
        return math.sqrt( self.distum() )
    pass

    def distum(self):
        p = self.p
        q = self.q
        return (p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y)
    pass
pass
