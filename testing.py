'''
testing.py: package containing necessary methods for testing add_new_data
    method for post-train data addition senior research
    
author: Jeremy Swerdlow

'''

''' ---------- imports ---------- '''

import data_prep as dp
import datetime as dt
import numpy as np
import pandas as pd

from tree import add_new_data, create_decision_tree, make_decision, InvalidPercentError

''' ---------- end imports ---------- '''


pet_df = dp.pets(tree=False, graph=False)
fv_day_df = dp.fv_day(tree=False, graph=False)
fv_day_2nd_df = dp.fv_day_2nd(tree=False, graph=False)
mushroom_df = dp.mushroom(tree=False, graph=False)


''' ---------- general testing methods ---------- '''

def metric_test(tree, df):
    tot = df.shape[0]
    tp = fp = tn = fn = 0
    pos, neg = tree.df[tree.target].unique().tolist()
    df = make_decision(df, tree, verbose=True)
    for _, row in df.iterrows():
        if row[tree.target] == pos:
            if row[tree.target] == row['results']:
                tp += 1
            else:
                fp += 1
        else:
            if row[tree.target] == row['results']:
                tn += 1
            else:
                fn += 1
    a = (tp + tn) / tot
    r = tp / float(tp + fn) if tp + fn != 0 else 0
    p = tp / float(tp + fp) if tp + fp != 0 else 0
    f = 2 * (p * r / (p + r)) if p + r != 0 else 0
    return a, r, p, f

def random_split_data(df, trn_pct, trn2_pct, tst_pct):
    df_len = df.shape[0]
    if trn_pct + trn2_pct + tst_pct > 1:
        raise(InvalidPercentError, "trn, trn2, and tst must total less than 1.")
    return np.split(df.sample(frac=1), [int(trn_pct*df_len), int((trn2_pct + trn_pct)*df_len)])

def df_from_results(test_res, time_dict, ttl):
    df = pd.DataFrame({k:v[ttl] for k, v in test_res.items()},
                      index=['accuracy', 'recall', 'precision', 'f-score']).T
    df['time'] = pd.DataFrame(time_dict).loc[ttl]
    return df

''' ---------- end general testing methods ---------- '''


''' ---------- random testing methods ---------- '''

def random_split_run():
    df_lbl_dict = {'pets':(pet_df, 'iscat'), 
                   'five_day':(fv_day_df, 'cats or dogs'),
                   'five_day_2':(fv_day_2nd_df, 'cats_or_dogs'),
                   'mushroom':(mushroom_df, 'class')}
    time_dict = {'initial':{},
                 'updated':{},
                 'remade':{}}
    test_res = {'initial':{},
                'updated':{},
                'remade':{}}
    for ttl, tpl in df_lbl_dict.items():
        df, cls = tpl
        trn, trn2, tst = random_split_data(df, .6, .2, .2)
        attrs = df.columns.tolist()
        attrs.remove(cls)

        # create the initial tree, with timing
        initial_start = dt.datetime.now()
        initial_tree = create_decision_tree(trn, attrs, cls, trn)
        initial_tree_time = dt.datetime.now() - initial_start
        time_dict['initial'][ttl] = initial_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(initial_tree, tst)
        test_res['initial'][ttl] = [a, r, p, f]


        # create the updated tree, with timing
        updated_start = dt.datetime.now()
        updated_tree = add_new_data(trn2, initial_tree)
        updated_tree_time = dt.datetime.now() - updated_start
        time_dict['updated'][ttl] = updated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(updated_tree, tst)
        test_res['updated'][ttl] = [a, r, p, f]


        # create the tree from full dataset, with timing
        full_trn = trn.append(trn2)
        recreated_start = dt.datetime.now()
        recreated_tree = create_decision_tree(full_trn, attrs, cls, full_trn)
        recreated_tree_time = dt.datetime.now() - recreated_start
        time_dict['remade'][ttl] = recreated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(recreated_tree, tst)
        test_res['remade'][ttl] = [a, r, p, f]
    
    return time_dict, test_res

''' ---------- end random testing methods ---------- '''

''' ---------- new node testing methods ---------- '''

def new_node_run():
    fv_col = 'Do you have any previous experience with programming?'
    fv_2_col = 'previous_programming_experience'
    df_lbl_dict = {'pets':(pet_df[pet_df['size'] != 'enormous'], 'iscat'), 
                   'five_day':(fv_day_df[fv_day_df[fv_col] != 'Nope'], 'cats or dogs'),
                   'five_day_2':(fv_day_2nd_df[fv_day_2nd_df[fv_2_col] != 'Nope'], 'cats_or_dogs'),
                   'mushroom':(mushroom_df[mushroom_df['odor'] != 'p'], 'class')}
    time_dict = {'initial':{},
                 'updated':{},
                 'remade':{}}
    test_res = {'initial':{},
                'updated':{},
                'remade':{}}
    
    for ttl, tpl in df_lbl_dict.items():
        df, cls = tpl
        trn, trn2, tst = random_split_data(df, .6, .2, .2)
        if ttl == 'pets':
            trn2.append(pet_df[pet_df['size'] == 'enormous'])
            tst.append(pet_df[pet_df['size'] == 'enormous'])
            
        elif ttl == 'five_day':
            targ_col = 'Do you have any previous experience with programming?'
            trn2.append(fv_day_df[fv_day_df[targ_col] == 'Nope'].iloc[:14])
            tst.append(fv_day_df[fv_day_df[targ_col] == 'Nope'].iloc[14:])
            
        elif ttl == 'five_day_2':
            targ_col = 'previous_programming_experience'
            trn2.append(fv_day_2nd_df[fv_day_2nd_df[targ_col] == 'Nope'].iloc[:17])
            tst.append(fv_day_2nd_df[fv_day_2nd_df[targ_col] == 'Nope'].iloc[17:])
            
        elif ttl == 'mushroom':
            trn2.append(mushroom_df[mushroom_df['odor'] == 'p'].iloc[:128])
            tst.append(mushroom_df[mushroom_df['odor'] == 'p'].iloc[128:])
            
        attrs = df.columns.tolist()
        attrs.remove(cls)

        # create the initial tree, with timing
        initial_start = dt.datetime.now()
        initial_tree = create_decision_tree(trn, attrs, cls, trn)
        initial_tree_time = dt.datetime.now() - initial_start
        time_dict['initial'][ttl] = initial_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(initial_tree, tst)
        test_res['initial'][ttl] = [a, r, p, f]


        # create the updated tree, with timing
        updated_start = dt.datetime.now()
        updated_tree = add_new_data(trn2, initial_tree)
        updated_tree_time = dt.datetime.now() - updated_start
        time_dict['updated'][ttl] = updated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(updated_tree, tst)
        test_res['updated'][ttl] = [a, r, p, f]


        # create the tree from full dataset, with timing
        full_trn = trn.append(trn2)
        recreated_start = dt.datetime.now()
        recreated_tree = create_decision_tree(full_trn, attrs, cls, full_trn)
        recreated_tree_time = dt.datetime.now() - recreated_start
        time_dict['remade'][ttl] = recreated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(recreated_tree, tst)
        test_res['remade'][ttl] = [a, r, p, f]
    
    return time_dict, test_res

''' ---------- end new node testing methods ---------- '''


''' ---------- new data testing methods ---------- '''

def new_data_run():
    fv_col = 'Do you have any previous experience with programming?'
    fv_2_col = 'previous_programming_experience'
    df_lbl_dict = {'pets':(pet_df[pet_df['size'] != 'enormous'], 'iscat'), 
                   'five_day':(fv_day_df[fv_day_df[fv_col] != 'Nope'], 'cats or dogs'),
                   'five_day_2':(fv_day_2nd_df[fv_day_2nd_df[fv_2_col] != 'Nope'], 'cats_or_dogs'),
                   'mushroom':(mushroom_df[mushroom_df['odor'] != 'p'], 'class')}
    time_dict = {'initial':{},
                 'updated':{},
                 'remade':{}}
    test_res = {'initial':{},
                'updated':{},
                'remade':{}}
    
    for ttl, tpl in df_lbl_dict.items():
        df, cls = tpl
        trn, trn2, tst = random_split_data(df, .6, .2, .2)
        if ttl == 'pets':
            trn.append(pet_df[pet_df['size'] == 'enormous'])
            trn2.append(pet_df[pet_df['size'] == 'enormous'])
            tst.append(pet_df[pet_df['size'] == 'enormous'])
            
        elif ttl == 'five_day':
            targ_col = 'Do you have any previous experience with programming?'
            trn.append(fv_day_df[fv_day_df[targ_col] == 'Nope'].iloc[:9])
            trn2.append(fv_day_df[fv_day_df[targ_col] == 'Nope'].iloc[9:18])
            tst.append(fv_day_df[fv_day_df[targ_col] == 'Nope'].iloc[18:])
            
        elif ttl == 'five_day_2':
            targ_col = 'previous_programming_experience'
            trn.append(fv_day_2nd_df[fv_day_2nd_df[targ_col] == 'Nope'].iloc[:11])
            trn2.append(fv_day_2nd_df[fv_day_2nd_df[targ_col] == 'Nope'].iloc[11:22])
            tst.append(fv_day_2nd_df[fv_day_2nd_df[targ_col] == 'Nope'].iloc[22:])
            
        elif ttl == 'mushroom':
            trn.append(mushroom_df[mushroom_df['odor'] == 'p'].iloc[:85])
            trn2.append(mushroom_df[mushroom_df['odor'] == 'p'].iloc[85:170])
            tst.append(mushroom_df[mushroom_df['odor'] == 'p'].iloc[170:])
            
        attrs = df.columns.tolist()
        attrs.remove(cls)

        # create the initial tree, with timing
        initial_start = dt.datetime.now()
        initial_tree = create_decision_tree(trn, attrs, cls, trn)
        initial_tree_time = dt.datetime.now() - initial_start
        time_dict['initial'][ttl] = initial_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(initial_tree, tst)
        test_res['initial'][ttl] = [a, r, p, f]


        # create the updated tree, with timing
        updated_start = dt.datetime.now()
        updated_tree = add_new_data(trn2, initial_tree)
        updated_tree_time = dt.datetime.now() - updated_start
        time_dict['updated'][ttl] = updated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(updated_tree, tst)
        test_res['updated'][ttl] = [a, r, p, f]


        # create the tree from full dataset, with timing
        full_trn = trn.append(trn2)
        recreated_start = dt.datetime.now()
        recreated_tree = create_decision_tree(full_trn, attrs, cls, full_trn)
        recreated_tree_time = dt.datetime.now() - recreated_start
        time_dict['remade'][ttl] = recreated_tree_time.total_seconds()

        # test it against metrics
        a, r, p, f = metric_test(recreated_tree, tst)
        test_res['remade'][ttl] = [a, r, p, f]
    
    return time_dict, test_res

''' ---------- end new data testing methods ---------- '''