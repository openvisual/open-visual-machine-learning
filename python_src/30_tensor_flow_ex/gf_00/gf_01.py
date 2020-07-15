# coding: utf-8

#conda install graphviz python-graphviz
#conda install tfgraphviz

from graphviz import Digraph

g = Digraph('G', )

g.edge('Hello', 'World')

g.view()