#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:54:47 2024

@author: yutong
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
import csv


''' 1. Read .csv files '''

my_path = os.path.abspath('/Users/yutong/pyEIT-results/r0.15')

file_n = 16
vf_lines = 64
cy_min = -0.90
cy_max = -0.60
cy_step = (cy_max-cy_min)/(file_n-1)

vf_array = np.zeros([file_n,vf_lines])

for i in range(0,file_n,1):
    cy = cy_min + cy_step * i 
    data_in = genfromtxt(my_path+'/vf_h0.015_cx0.00_cy%.2f_r0.15_perm100.csv'% cy, delimiter=',')
    vf_array[i,:] = data_in[1:(vf_lines+1),1] 
    data_in = None
    del data_in
    
    
''' 2. Find potential difference between electrode #1 and #4 '''

vdiff_14 = np.zeros([file_n])

for i in range(0,file_n,1):
    vdiff_14[i] = vf_array[i,:].max() - vf_array[i,:].min()
    

''' 3. Find potential difference between electrode #2 and #3 '''

vdiff_23 = np.zeros([file_n])

# Need to manually input the difference of vlines

line_diff = np.array([0.6,
0.8,
0.9,
1,
1.2,
1.4,
1.6,
1.8,
2,
2.1,
2.2,
2.4,
2.5,
2.6,
2.8,
3])

line_diff = line_diff*2 + 1

for i in range(0,file_n,1):
    line_step = (vf_array[i,:].max() - vf_array[i,:].min())/(vf_lines-1)
    vdiff_23[i] = line_step * line_diff[i]
    
''' 4. Plot '''

dist = np.linspace(cy_min, cy_max, file_n) + 0.9

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Distance from electrode plane (normalized M)')
ax1.set_ylabel('diff_23 (normalized V)', color=color)
ax1.plot(dist, vdiff_23, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('diff_14 (normalized V)', color=color)  # we already handled the x-label with ax1
ax2.plot(dist, vdiff_14, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()










