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

        self.lineListIdentified = None
    pass # -- __init__

    def identify(self, lineListB, snapDeg=10, snapDistRatio=0.1):
        fileBase = self.fileBase
        w = self.w
        h = self.h
        algorithm = self.algorithm

        lines_identified = []

        lineListA = self

        linesA = lineListA.lines

        for line in linesA :
            line_identified = line.get_identified_line( lineListB, snapDeg=snapDeg, snapDistRatio=snapDistRatio )
            if line_identified :
                line.line_identified = line_identified

                lines_identified.append( line )
            pass
        pass

        lineList = LineList( lines = lines_identified, algorithm=algorithm, w=w, h= h, fileBase=fileBase)

        return lineList
    pass # -- identify

    def save_as_json(self, json_file_name ):
        pass
    pass # -- save_as_json
pass # -- LineList
