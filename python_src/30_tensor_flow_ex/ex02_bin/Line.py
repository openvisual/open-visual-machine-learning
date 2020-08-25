# encoding: utf-8

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y =
    pass
pass

class Line:
    def __init__(self, line):
        self.a = Point(line[0], line[1])
        self.b = Point(line[2], line[3])
    pass

    def __init__(self,a : Point ,b : Point ):
        self.a=a
        self.b=b
    pass

    def distum(self):
        a = self.a
        b = self.b
        return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)
    pass
pass
