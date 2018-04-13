'''
tree.py: package containing methods related to the creation, editing, and
    editing of decision trees for the post-train data addition senior research
    
author: Jeremy Swerdlow

'''

''' ---------- imports ---------- '''

import math as m

from copy import deepcopy
from six import string_types

''' ---------- end imports ---------- '''


''' ---------- implemented Errors ---------- '''

class DifferentColumnsError(Exception):
    pass

class HitStringError(Exception):
    pass

class InvalidPercentError(Exception):
    pass

''' ---------- end implemeneted Errors ---------- '''


''' ---------- class and dependent functions for TreeNode ---------- '''

class TreeNode:
    '''
    TreeNode - class definition for the nodes of the decision tree
        stores: 
            - the decision
            - the pd dataframe at that decision
            - the target column name
            - the children nodes as values of dictionary 
                where key is the split attribute
    '''
    def __init__(self, decision, df, target, rem_attributes):
        self.decision = decision
        self.target = target
        self.df = df
        self.attributes_left = rem_attributes
        self.children = {}

def print_tree(tree_obj):
    '''
    print_tree - function to print out an object of TreeNode type
    '''
    def _print_tree(tree_obj, tab_cnt):
        tab_delim = ' |   '
        if not isinstance(tree_obj, string_types):
            for k, v in tree_obj.children.items():
                if isinstance(v, string_types):
                    print(tab_delim * tab_cnt, 
                          tree_obj.decision, ' = ', k, ': ', v)
                else:
                    print(tab_delim * tab_cnt, tree_obj.decision, ' = ', k)
                    _print_tree(v, tab_cnt + 1)
        else:
            print(tab_delim * tab_cnt, tree_obj)
    _print_tree(tree_obj, 0)

''' ---------- end of class TreeNode and related methods ---------- '''


''' ---------- Methods used to create most_gain evaluation fxn ---------- '''

def entropy(col):
    def _entropy(val):
        if val == 0:
            return 0
        else:
            return val * m.log(val, 2)
        
    vals = list(col.value_counts())
    total = len(col)
    vals = list(map(lambda x: x/total, vals))
    vals = list(map(_entropy, vals))
    return -1 * sum(vals)

def calculate_probability(df, attribute, v):
    return len(df[df[attribute] == v]) / len(df)
    
def remainder(df, attribute, targ_attr):
    rem = 0
    for v in df[attribute].unique():
        v_p = calculate_probability(df, attribute, v)
        v_e = entropy(df[df[attribute] == v][targ_attr])
        rem += v_p * v_e
    return rem

def gain(df, attribute, targ_attr):
    return entropy(df[targ_attr]) - remainder(df, attribute, targ_attr)

def most_gain(df, attributes, targ_attr):
    best_attr = attributes[0]
    best_gain = 0
    for a in attributes:
        a_gain = gain(df, a, targ_attr)
        if a_gain > best_gain:
            best_attr = a
            best_gain = a_gain
    return best_attr, best_gain

''' ---------- end of most_gain and its dependencies ---------- '''


''' ---------- estimation functions ---------- '''

def majority_val(df, targ_attr):
    return df[targ_attr].value_counts().index[0]

''' ---------- end estimation functions ---------- '''


'''---------- create_decision_tree method ----------'''

def create_decision_tree(df, attributes, targ_attr, parent_df, 
                         eval_fxn=most_gain, est_fxn=majority_val):
    '''
    create_decision_tree - function to create a decision tree based on
        splits decided by eval_fxn. built on pandas dataframes
    '''
    if df.empty:
        return est_fxn(p_df, targ_attr)
    elif df[targ_attr].value_counts().iloc[0] == len(df[targ_attr]):
        return df[targ_attr].iloc[0]
    elif attributes == []:
        return est_fxn(parent_df, targ_attr)
    else:
        decision, gain = eval_fxn(df, attributes, targ_attr)
        if gain == 0:
            return est_fxn(parent_df, targ_attr)
        rem_attributes = attributes[:]
        rem_attributes.remove(decision)
        node = TreeNode(decision, df, targ_attr, rem_attributes)
        for val in df[decision].unique():
            exs = df[df[decision] == val]
            node.children[val] = create_decision_tree(exs, rem_attributes, 
                                                      targ_attr, df, 
                                                      eval_fxn=eval_fxn,
                                                      est_fxn=est_fxn)
        return node

'''---------- end create_decision_tree method ----------'''


''' ---------- make_decision method ---------- '''

def make_decision(df, tree, est_fxn=majority_val, verbose=True):
    '''
    make_decision - function to make a decision for data from df based on
        tree's structure, and the estimation function defined in est_fxn
    '''
    def _make_decision(row, tree, est_fxn=majority_val):
        if isinstance(tree, string_types):
            return tree
        elif row[tree.decision] in tree.children.keys():
            return _make_decision(row, 
                                  tree.children[row[tree.decision]], 
                                  est_fxn=est_fxn)
        else:
            return est_fxn(tree.df, tree.target)
    results = []
    for _, row in df.iterrows():
        results.append(_make_decision(row, tree, est_fxn=est_fxn))
    
    if verbose:
        df['results'] = results
        return df
    else:
        return results

''' ---------- end make_decision method ---------- '''


'''---------- add_new_data method ----------'''

def add_new_data(df, tree, eval_fxn=most_gain, est_fxn=majority_val, copy=False):
    '''
    add_new_data - takes a dataframe of labelled samples, and adds them to tree,
        based on eval_fxn and est_fxn
    '''
    def add_new_row(row, tree, eval_fxn=most_gain, est_fxn=majority_val):
        tree.df = tree.df.append(row, ignore_index=True)
        dec = tree.decision
        if row[dec] in tree.children.keys():
            if isinstance(tree.children[row[dec]], string_types):
                if row[tree.target] == tree.children[row[dec]]:
                    pass
                else:
                    # see if there's a new way to split data further
                    exs = tree.df[tree.df[dec] == row[dec]]
                    tree.children[row[dec]] = create_decision_tree(exs, tree.attributes_left,
                                                                   tree.target, tree.df,
                                                                   eval_fxn=eval_fxn,
                                                                   est_fxn=est_fxn)
            else:
                # recurse to child
                tree.children[row[dec]] = add_new_row(row, tree.children[row[dec]], 
                                                      eval_fxn=eval_fxn, est_fxn=est_fxn)
        else:
            # only one child, set its value as target
            tree.children[row[dec]] = row[tree.target]
        return tree
    
    if copy:
        tree = deepcopy(tree)
    
    if len(df.columns) != len(tree.df.columns):
        raise DifferentColumnsError('new data has fewer columns than TreeNode.df')
    if not all(df.columns == tree.df.columns):
        raise DifferentColumnsError('new data different columns than TreeNode.df')

    for _, row in df.iterrows():
        tree = add_new_row(row, tree, eval_fxn=most_gain, est_fxn=majority_val)
        
    return tree

''' ---------- end add_new_data method ---------- '''


'''---------- drop_non_categorical method ----------'''

def drop_non_categorical(df):
    for col in df.columns.tolist():
        if df[col].dtype in [int, float, complex]:
            df = df.drop(col, axis=1)
    return df

'''---------- end drop_non_categorical method ----------'''