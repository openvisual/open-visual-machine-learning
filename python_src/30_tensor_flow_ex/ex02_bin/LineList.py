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

    def line_identify(self, lineListB, snapDeg=10, snapDistRatio=0.1):
        fileBase = self.fileBase
        w = self.w
        h = self.h
        algorithm = self.algorithm

        lines_identified = []

        for line in self.lines :
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
        import json
        #data = {'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'}
        data = {}

        lines = self.lines

        for i, line in enumerate( lines ) :
            line_a = line
            line_b = line.line_identified

            line_data = {}

            line_data[line_a.fileBase] = {"point1": [line_a.a.x, line_a.a.y], "point2": [line_a.b.x, line_a.b.y]}
            line_data[line_b.fileBase] = {"point1": [line_b.a.x, line_b.a.y], "point2": [line_b.b.x, line_b.b.y]}

            data[ f"line{i +1}" ] = line_data
        pass

        with open( json_file_name, 'w') as f:
            json.dump(data, f, indent=4)
        pass

    pass # -- save_as_json
pass # -- LineList
