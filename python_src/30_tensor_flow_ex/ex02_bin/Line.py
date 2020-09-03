# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from functools import cmp_to_key

import math

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    pass

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return None
        pass

    pass

    def __str__(self):
        return f"Point( {self.x}, {self.y} )"

    pass

    def distum(self, p ):
        dx = self.x - p.x
        dy = self.y - p.y
        return dx*dx + dy*dy
    pass

    def distance(self, p):
        return math.sqrt(self.distum(p))
    pass

    @staticmethod
    def compare_point_x(a, b):
        return a.x - b.x
    pass

pass # -- Point

class Line:

    ID = 0

    def __init__(self, a=None , b=None, line=None ):
        self.id = Line.ID
        Line.ID += 1

        if line is not None :
            self.a = Point(line[0], line[1])
            self.b = Point(line[2], line[3])
        else :
            self.a = a
            self.b = b
        pass
    pass

    def __getitem__(self, i):
        if i == 0 :
            return self.a
        elif i == 1 :
            return self.b
        else :
            return None
        pass
    pass

    def __str__(self):
        return f"Line({self.a.x}, {self.a.y}, {self.b.x}, {self.b.y})"
    pass

    def length(self):
        return math.sqrt( self.distum() )
    pass

    def distum(self):
        return self.a.distum(self.b)
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
        return self.b.x - self.a.x
    pass

    def dy(self):
        return self.b.y - self.a.y
    pass

    def slope_radian(self):
        rad = math.atan2( self.dy() , self.dx() )
        rad = rad % (2*math.pi)

        return rad
    pass

    def is_same_slope(self, line, error_deg=1 ):
        sa = math.degrees( self.slope_radian() )
        sb = math.degrees( line.slope_radian() )

        diff_deg = abs( sa - sb )

        0 and log.info( f"sa = {sa:.2f}, sb = {sb:.2f}, diff = {diff_deg:.2f}")

        return diff_deg <= error_deg
    pass

    def merge(self, line, error_deg=1, snap_dist=5 ):
        merge_line = None
        debug = 0

        if self.is_mergeable( line , error_deg=error_deg, snap_dist=snap_dist) :
            points = [self.a, self.b, line.a, line.b]

            debug and log.info( f"points org = { ', '.join([str(p) for p in points]) }")

            points = sorted(points, key=cmp_to_key(Point.compare_point_x))

            debug and log.info( f"points sort = { ', '.join([str(p) for p in points]) }")

            merge_line = Line( a = points[0], b = points[-1] )

            debug and log.info( f"merge line = {merge_line}")
        pass

        return merge_line
    pass

    def is_mergeable(self, line, error_deg, snap_dist ):
        if self.is_same_slope( line, error_deg = error_deg ) :
            from shapely.geometry import LineString

            line1 = LineString([(self.a.x, self.a.y), (self.b.x, self.b.y)])
            line2 = LineString([(line.a.x, line.a.y), (line.b.x, line.b.y)])

            dist = line1.distance(line2)

            0 and log.info( f"dist = {dist}" )

            return dist <= snap_dist
        else :
            return False
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

        line_merged = True

        while line_merged :
            line_merged = False

            i = 0
            lines = sorted(lines, key=cmp_to_key(Line.compare_line_slope))

            while i < len(lines) - 1 :
                j = 0
                while j < len(lines):
                    merge_line = None

                    if i is not j :
                        merge_line = lines[i].merge(lines[j], error_deg=error_deg, snap_dist=snap_dist)
                    pass

                    if merge_line is not None:
                        line_merged = True
                        lines[i] = merge_line
                        lines.pop(j)

                        log.info(f"Line({i}, {j}) are merged.")
                    else:
                        j += 1
                    pass
                pass

                i += 1
            pass
        pass

        return lines
    pass

pass # -- Line
