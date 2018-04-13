'''
graph_tree.py: package containing necessary classes and imports for graphing
    a decision tree from tree.py using the graphviz DOT language
    
author: Jeremy Swerdlow

'''

''' ---------- imports ---------- '''

from bokeh.palettes import viridis
from graphviz import Digraph
from six import string_types

''' ---------- end imports ---------- '''

''' ---------- color_handler class ---------- '''

class color_handler:
    '''
    color_handler - sets and gets colors for tree when graphing it, based
        on number of classes, structure, etc.
    '''
    def __init__(self, tree):
        targs = tree.df[tree.target].unique().tolist()
        num_colors = len(targs) if len(targs) >= 3 else 3
        colors = viridis(num_colors)
        self.colors = {targs[i]:colors[i] for i in range(len(targs))}
        
    @staticmethod
    def font_color(bg_hex):
        # built from response on https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
        rgb = [int(bg_hex[x:x + 2], 16) for x in [1, 3, 5]]
        for c in rgb:
            c = c / 255
            c = c / 12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4
        r, g, b = rgb[0], rgb[1], rgb[2]
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        color = '#000000' if L > 0.179 else '#ffffff'
        return color
        
    @staticmethod
    def weight_color(color, percent_increase):
        offset = int(256 * (percent_increase - 1))
        rgb = [color[x:x + 2] for x in [1, 3, 5]]
        rgb = [int(val, 16) + offset for val in rgb]
        rgb = [min(255, max(0, val)) for val in rgb]
        return '#' + ''.join([hex(val)[2:] for val in rgb])
    
    def get_color(self, tree):
        if isinstance(tree, string_types): # leaf node
            return self.colors[tree]
        
        targs = tree.df[tree.target].value_counts()
        targ = targs.index[0]
        color = self.colors[targ]
        
        color_weight = targs[targ] / sum(targs)
        percent = 1 / color_weight
        
        return color_handler.weight_color(color, percent)

def graph_tree(tree, title=None, comment=None, file_name=None):
    '''
    graph_tree - function to create a graphviz object from a decision tree
    '''
    def _graph_tree(graph, tree, parent_str, counter, target):
        for edge_lbl, child in tree.children.items():
            e_lbl = str(edge_lbl)
            if isinstance(child, string_types):
                graph.attr('node', fillcolor=c.get_color(child))
                graph.attr('node', fontcolor=color_handler.font_color(c.get_color(child)))
                graph.node(str(counter), target + ": \n" + child)
                graph.edge(parent_str, str(counter), e_lbl)
                counter += 1
            else:
                c_str = child.__str__()
                graph.attr('node', fillcolor=c.get_color(child))
                graph.attr('node', fontcolor=color_handler.font_color(c.get_color(child)))
                graph.node(c_str, child.decision + ': \n' + str(dict(child.df[child.target].value_counts())))
                graph.edge(parent_str, c_str, e_lbl)
                counter = _graph_tree(graph, child, c_str, counter, target)
        return counter
    g = Digraph(name=title, comment=comment, filename=file_name, format='png')
    g.attr('node', style='rounded,filled', shape='box')
    c = color_handler(tree)
    p_str = tree.__str__()
    root = tree
    g.node(p_str, tree.decision)
    _graph_tree(g, tree, p_str, 0, tree.target)
    return g

'''---------- end graphing methods ----------'''