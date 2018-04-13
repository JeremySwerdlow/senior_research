'''
data_prep.py: file to handle the importing of datasets for the post-train
    data addition senior research

author: Jeremy Swerdlow

'''

''' ---------- begin imports ----------'''

import pandas as pd

from graph_tree import graph_tree
from tree import create_decision_tree, drop_non_categorical

''' ---------- end methods ----------'''


''' ---------- begin methods ----------'''

def pets(tree=True, graph=True):
    '''
    returns a 3-tuple of the dataframe, tree, and graph for the pets dataset
    
    can selectively return only the dataframe, only the dataframe and tree,
        or all three by setting tree and graph.
    '''
    pet_df = pd.read_csv('datasets/pets.txt', '\t')
    if tree:
        pet_tree = create_decision_tree(pet_df,
                                        list(pet_df.columns[:-1]), 
                                        'iscat', 
                                        pet_df)
        if graph:
            pet_graph = graph_tree(pet_tree)
            return pet_df, pet_tree, pet_graph
        else:
            return pet_df, pet_tree
    else:
        return pet_df

folder = 'datasets/5day-data-challenge-signup-survey-responses/'

def fv_day(tree=True, graph=True):
    '''
    returns a 3-tuple of the dataframe, tree, and graph for the five-day data challenge
        first dataset
    
    can selectively return only the dataframe, only the dataframe and tree,
        or all three by setting tree and graph.
    '''
    long_class = 'Just for fun, do you prefer dogs or cat?'
    fv_day_df = pd.read_csv(folder + 'anonymous-survey-responses.csv')
    fv_day_df['cats or dogs'] = list(map(lambda x: x[:-3], fv_day_df[long_class]))
    fv_day_df = fv_day_df.drop(long_class, axis=1)
    fv_day_df = fv_day_df[fv_day_df['cats or dogs'] != 'Both ?']
    fv_day_df = fv_day_df[fv_day_df['cats or dogs'] != 'Neither']
    if tree:
        fv_day_tree = create_decision_tree(fv_day_df.head(len(fv_day_df) - 2), 
                                           list(fv_day_df.columns[:-1]), 
                                           'cats or dogs', 
                                           fv_day_df.head(len(fv_day_df) - 2))
        if graph:
            fv_day_grph = graph_tree(fv_day_tree)
            return fv_day_df, fv_day_tree, fv_day_grph
        else:
            return fv_day_df, fv_day_tree
    else:
        return fv_day_df

def fv_day_2nd(tree=True, graph=True):
    '''
    returns a 3-tuple of the dataframe, tree, and graph for the five-day data challenge
        second dataset
    
    can selectively return only the dataframe, only the dataframe and tree,
        or all three by setting tree and graph.
    '''
    
    fv_day_2nd_df = pd.read_csv(folder + 'anonymous-survey-responses-2nd-challenge.csv')
    fv_day_2nd_df = fv_day_2nd_df.dropna()
    fv_day_2nd_df = fv_day_2nd_df[fv_day_2nd_df['cats_or_dogs'] != 'Neither ']
    fv_day_2nd_df = fv_day_2nd_df[fv_day_2nd_df['cats_or_dogs'] != 'Both ']
    fv_day_2nd_df = drop_non_categorical(fv_day_2nd_df)
    fv_day_2nd_df.reset_index(drop=True, inplace=True)
    if tree:
        fv_day_2nd_tree = create_decision_tree(fv_day_2nd_df, 
                                               fv_day_2nd_df.columns.tolist()[:-1], 
                                               'cats_or_dogs', 
                                               fv_day_2nd_df)
        if graph:
            fv_day_2nd_grph = graph_tree(fv_day_2nd_tree)
            return fv_day_2nd_df, fv_day_2nd_tree, fv_day_2nd_grph
        else:
            return fv_day_2nd_df, fv_day_2nd_tree
    else:
        return fv_day_2nd_df
    
def mushroom(tree=True, graph=True):
    '''
    returns a 3-tuple of the dataframe, tree, and graph for the mushroom data
    
    can selectively return only the dataframe, only the dataframe and tree,
        or all three by setting tree and graph.
    '''
    mushroom_df = pd.read_csv('datasets/mushrooms.csv')
    mushroom_df['class'] = mushroom_df['class'].map({'e':'edible', 'p':'poisonous'})
    if tree:
        mushroom_tree = create_decision_tree(mushroom_df, 
                                             mushroom_df.columns.tolist()[1:], 
                                             'class', 
                                             mushroom_df)
        if graph:
            mushroom_graph = graph_tree(mushroom_tree)
            return mushroom_df, mushroom_tree, mushroom_graph
        else:
            return mushroom_df, mushroom_tree
    else:
        return mushroom_df
    
''' ---------- end methods ----------'''