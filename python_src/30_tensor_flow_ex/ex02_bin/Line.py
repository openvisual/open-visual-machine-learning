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

    def thickness(self):
        length = self.length()

        if length > 1000:
            thickness = 23 + length / 1000
        elif length > 100:
            thickness = 12 + length / 100
        else:
            thickness = 2 + length / 10
        pass

        thickness = int(thickness)

        return thickness
    pass

    @staticmethod
    def compare_line_length(a, b):
        return a.distum() - b.distum()
    pass

pass
