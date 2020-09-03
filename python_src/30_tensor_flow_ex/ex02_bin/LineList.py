# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

class LineList :
    def __init__(self, lines=[], algorithm="", w = 0 , h = 0, img_path="", mode=""):
        self.mode = mode
        self.lines = lines
        self.algorithm = algorithm

        self.w = w
        self.h = h
        self.img_path = img_path
    pass
pass # -- LineList
