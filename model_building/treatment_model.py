#impulse treatment model

import numpy as np
from get_tumor_db import get_tumor_db
from parse_tumor_db import parse_tumor_db

def get_tt(tuple_treatment_group):
    #treatment times
    switcher={
        ('dox',1): [39],
        
        ('her',2): [35,38],
        
        ('dox',3): [35],
        ('her',3): [36,39],
        
        ('her',4): [35,38],
        ('dox',4): [39],
        
        ('her',5): [35,38],
        ('dox',5): [35,38]
    }  
    return switcher.get(tuple_treatment_group, [])

#Return treatment for all groups
def get_Sd_impulse(ts):

    Sds = np.zeros((6, len(ts)))
    for group in range(6):
        treatment_times = get_tt(('dox', group))
        for tt in treatment_times:
            ii = np.where(ts > tt)[0][0]
            Sds[group, ii] = 1
    return Sds
    
def get_Sh_impulse(ts):
    #returns Herceptin treatment for all 6 groups
    Shs = np.zeros((6, len(ts)))
    for group in range(6):
        treatment_times = get_tt(('her', group))
        for tt in treatment_times:
            ii = np.where(ts > tt)[0][0]
            Shs[group, ii] = 1
    return Shs

#return treatment for 1 group
def get_Sd_impulse_by_group(ts, group):
    Sds = np.zeros((1, len(ts)))
    treatment_times = get_tt(('dox', group))
    for tt in treatment_times:
        ii = np.where(ts > tt)[0][0]
        Sds[0, ii] = 1
    return Sds
    
def get_Sh_impulse_by_group(ts, group):
    Shs = np.zeros((1, len(ts)))
    treatment_times = get_tt(('her', group))
    for tt in treatment_times:
        ii = np.where(ts > tt)[0][0]
        Shs[0, ii] = 1
    return Shs

#return treatment for combination of groups
def get_Sd_impulse_Combination(ts, group_idxs):
    Sds = np.zeros((len(group_idxs), len(ts)))
    for ii in range(len(group_idxs)):
        group_idx = group_idxs[ii]
        treatment_times = get_tt(('dox', group_idx))
        for tt in treatment_times:
            jj = np.where(ts > tt)[0][0]
            Sds[ii, jj] = 1
    return Sds

def get_Sh_impulse_Combination(ts, group_idxs):
    Shs = np.zeros((len(group_idxs), len(ts)))
    for ii in range(len(group_idxs)):
        group_idx = group_idxs[ii]
        treatment_times = get_tt(('her', group_idx))
        for tt in treatment_times:
            jj = np.where(ts > tt)[0][0]
            Shs[ii, jj] = 1
    return Shs



#tests

import matplotlib.pyplot as plt

tumor_size_db = get_tumor_db()
ts, Ts, sigmas = parse_tumor_db(tumor_size_db)

def test_treatment_Sd_impulse():
    tau_d=0.1
    Sds = get_Sd_impulse(ts)
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    for group in range(6):
        axs[int(group/3), int(group%3)].title.set_text('Dox Group ' + str(group))
        axs[int(group/3), int(group%3)].stem(ts, Sds[group], use_line_collection=True)
    plt.show()
    return

def test_treatment_Sh_impulse():
    Shs = get_Sh_impulse(ts)
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    for group in range(6):
        axs[int(group/3), int(group%3)].title.set_text('Her Group ' + str(group))
        axs[int(group/3), int(group%3)].stem(ts, Shs[group], use_line_collection=True)
    plt.show()
    return


def test_treatment_Sd_impulse_by_group():
    Sds = np.zeros((6, len(ts)))
    for group in range(6):
        Sds[group,:] = get_Sd_impulse_by_group(ts, group)
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    for group in range(6):
        axs[int(group/3), int(group%3)].title.set_text('Dox Group ' + str(group))
        axs[int(group/3), int(group%3)].stem(ts, Sds[group], use_line_collection=True)
    plt.show()
    return


def test_treatment_Sh_impulse_by_group():
    #ts = np.array([int(x) for x in np.linspace(0,99,100)])
    Ts = np.ones(ts.shape)
    Shs = np.zeros((6, len(ts)))
    for group in range(6):
        Shs[group,:] = get_Sh_impulse_by_group(ts, group)
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    for group in range(6):
        axs[int(group/3), int(group%3)].title.set_text('Her Group ' + str(group))
        axs[int(group/3), int(group%3)].stem(ts, Shs[group], use_line_collection=True)
    plt.show()
    return


print('teatment impulse')
test_treatment_Sd_impulse()
test_treatment_Sh_impulse()

#print('treatment impulse by group')
#test_treatment_Sd_impulse_by_group()
#test_treatment_Sh_impulse_by_group()
