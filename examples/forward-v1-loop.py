#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:47 2024

@author: yutong
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.interp2d import sim2pts, pdegrad

my_path = os.path.abspath('/Users/yutong/pyEIT-results/')

""" 0. build mesh """
n_el = 4  # nb of electrodes
use_customize_shape = True

h0_def = 0.015

if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=h0_def, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.03)

el_pos = mesh_obj.el_pos

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]
mesh_obj.print_stats()


# change permittivity
cx_def = 0
cy_def = -0.9
r_def = 0.15
perm_def = 100

while (cy_def <= -0.8):
    
    center_def = [cx_def,cy_def]
    anomaly = PyEITAnomaly_Circle(center=center_def, r=r_def, perm=perm_def)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new.perm
    
    """ 1. FEM forward simulations """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=7, step_meas=1, parser_meas="std")
    
    # Define electrode current sink and current source
    ex_mat = protocol.build_exc_pattern_std(dist=3)
    ex_line = protocol_obj.ex_mat[0].ravel()
    
    # calculate simulated data using FEM
    fwd = Forward(mesh_new)
    f = fwd.solve(ex_line)
    f = np.real(f)
    
    """ 2. plot """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 64)
    
    # vf = np.sort(f[el_pos])
    # Draw contour lines on an unstructured triangular grid.
    ax[0].tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
    
    # draw mesh structure
    # Create a pseudocolor plot of an unstructured triangular grid
    ax[0].tripcolor(
        x,
        y,
        tri,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.05,
        cmap=plt.cm.Greys,
    )
    
    # draw electrodes
    ax[0].plot(x[el_pos], y[el_pos], "ro")
    for i, e in enumerate(el_pos):
        ax[0].text(x[e], y[e], str(i + 1), size=12)
    ax[0].set_title("Equipotential Lines")
    # clean up
    ax[0].set_aspect("equal")
    ax[0].set_ylim([-1, 0.2])
    ax[0].set_xlim([-1, 1])
    fig.set_size_inches(6, 6)
    
    ux, uy = pdegrad(pts, tri, f)
    uf = ux**2 + uy**2
    uf_pts = sim2pts(pts, tri, uf)
    uf_logpwr = 10 * np.log10(uf_pts)
    
    # Draw contour lines on an unstructured triangular grid.
    ax[1].tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.viridis)
    ax[1].tricontour(x, y, tri, uf_logpwr, 10, cmap=plt.cm.hot)
    ax[1].set_aspect("equal")
    ax[1].set_ylim([-1, 0.2])
    ax[1].set_xlim([-1, 1])
    fig.set_size_inches(6, 6)
    ax[1].set_title("Electric Field")
    # plt.show()
    
    fig.set_size_inches(16, 8)
    fig.savefig(my_path + 'sim_h%.3f_cx%.2f_cy%.2f_r%.2f_perm%.0f.png' % (h0_def, cx_def, cy_def, r_def, perm_def), dpi=800)
    
    # """ 3. calculate potential difference """
    
    # vdiff_23 = vf[18] - vf[17]
    # vdiff_14 = max(vf) - min(vf)
    
    # file1 = open("results_ywu/data_h%.3f_cx%.2f_cy%.2f_r%.2f_perm%.0f.txt" % (h0_def, cx_def, cy_def, r_def, perm_def), "w") 
    # str_vdiff_23 = '%f' % vdiff_23
    # str_vdiff_14 = '%f' % vdiff_14
    # L = [str_vdiff_23, '\n', str_vdiff_14]
    # file1.writelines(L)
    # file1.close()
    
    dvf = pd.DataFrame(vf)
    dvf.to_csv(my_path + '/vf_h%.3f_cx%.2f_cy%.2f_r%.2f_perm%.0f.csv' % (h0_def, cx_def, cy_def, r_def, perm_def))


    cy_def = cy_def + 0.1


