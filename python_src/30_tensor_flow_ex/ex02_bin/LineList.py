# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import math

class LineList :
    def __init__(self, lines=[], algorithm="", w = 0 , h = 0, fileBase="", mode=""):
        self.mode = mode
        self.lines = lines
        self.algorithm = algorithm

        self.w = w
        self.h = h

        self.diagonal = math.sqrt(w * w + h * h)

        self.fileBase = fileBase
    pass # -- __init__

    def get_lines_under_snap_radian(self, refLine = None, snapRadian = None, snapDistRatio = 0.1):
        lines_filter = []

        lines = self.lines

        diagonal = self.diagonal

        ref_slope_rad = refLine.slope_radian()
        ref_line_len = refLine.length()

        two_pi = 2*math.pi

        for line in lines :
            slope_rad = line.slope_radian()
            diff_rad = abs(slope_rad - ref_slope_rad) % two_pi
            if diff_rad < snapRadian :
                line_len = line.length()
                diff_len_ratio = abs( ref_line_len - line_len )/max( [ref_line_len, line_len] )
                if diff_len_ratio < snapDistRatio :

                    lines_filter.append( line )
                pass
            pass
        pass

        return lines_filter
    pass # -- get_lines_under_snap_radian

    def merge(self, lineListB, snapDeg=10, snapDistRatio=0.1):
        fileBase = self.fileBase
        diagonal = self.diagonal
        w = self.w
        h = self.h
        algorithm = self.algorithm

        snapRad = (math.pi/180)*snapDeg

        snapDist = diagonal*snapDistRatio

        linesMerge = []

        lineList = LineList( lines = linesMerge , algorith=algorithm, w=w, h= h, fileBase=fileBase)

        return lineList
    pass # -- merge
pass # -- LineList
