# encoding: utf-8
import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from functools import cmp_to_key

import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    pass

    def distum(self, p ):
        return (self.x - p.x)*(self.x - p.x) + (self.y - p.y)*(self.y - p.y)
    pass

    def distance(self, p):
        return math.sqrt( self.distum(p))
    pass

    def __getitem__(self, i):
        if i == 0 :
            return self.x
        elif i == 1 :
            return self.y
        else :
            return None
        pass
    pass

    @staticmethod
    def compare_point_x(a, b):
        return a.x - b.x
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

    def __getitem__(self, i):
        if i == 0 :
            return self.p
        elif i == 1 :
            return self.q
        else :
            return None
        pass
    pass

    def length(self):
        return math.sqrt( self.distum() )
    pass

    def distum(self):
        return self.p.distum( self.q )
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

    def dx(self):
        return self.q.x - self.p.x
    pass

    def dy(self):
        return self.q.y - self.p.y
    pass

    def slope_radian(self):
        rad = math.atan2( self.dy() , self.dx() )
        rad = rad % (2*math.pi)

        return rad
    pass

    def is_same_slope(self, line, error_deg=1 ):
        diff_deg = abs( math.degrees(self.slope_radian() - line.slope_radian()) )

        return diff_deg <= error_deg
    pass

    def merge(self, line, error_deg=1, snap_dist=5 ):
        merge_line = None

        if self.is_same_slope( line , error_deg=error_deg) and self.distance(line) < snap_dist :
            points = [ self.p, self.q, line.p, line.q ]

            points = sorted(points, key=cmp_to_key(Point.compare_point_x))

            merge_line = Line( p = points[0], q = points[-1] )
        pass

        return merge_line
    pass

    def distance(self, line):
        if self.is_intersects( line ) :
            return 0
        pass

        p = self.p
        q = self.q

        distums = [ p.distum(line.p) , p.distum(line.q), q.distum( line.p), q.distum(line.q) ]

        min_distum = min( distums )

        return math.sqrt( min_distum )
    pass

    def is_intersects(self, line):
        if self.intersect_point(line) is not None:
            return True
        else:
            return False
        pass

    pass

    def my_line(self):
        p = self.p
        q = self.q

        a = (p[1] - q[1])
        b = (q[0] - p[0])
        c = (q[0] * p[1] - p[0] * q[1])

        return a, b, c
    pass

    def intersect_point(self, line ):
        l1 = self.my_line()
        l2 = line.my_line()

        d = l1[0] * l2[1] - l1[1] * l2[0]

        if d == 0 :
            return None
        else :
            dx = l1[2] * l2[1] - l1[1] * l2[2]
            dy = l1[0] * l2[2] - l1[2] * l2[0]

            x = dx / d
            y = dy / d

            return Point( x, y )
        pass
    pass

    @staticmethod
    def compare_line_length(a, b):
        return a.distum() - b.distum()
    pass

    @staticmethod
    def compare_line_slope(a, b):
        return a.slope_radian() - b.slope_radian()
    pass

    @staticmethod
    def merge_lines( lines, error_deg=1, snap_dist=5 ):
        lines = lines.copy()

        lines = sorted(lines, key=cmp_to_key(Line.compare_line_slope))

        i = 0
        while i < len(lines) - 1:
            line = lines[i]
            j = i + 1
            while j < len(lines) :
                line2 = lines[j]
                merge_line = line.merge(line2, error_deg=error_deg, snap_dist=snap_dist)
                if merge_line is not None:
                    lines[i] = merge_line
                    lines.pop( j )

                    log.info( f"Line({i}, {j}) are merged." )
                else :
                    j += 1
                pass
            pass
            i += 1
        pass

        return lines
    pass

pass
