#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:33:46 2021

This script creates Fig. 12 of the paper from the estimated thresholds and evaluated capacities.
@author: Benedikt Vettelschoss
"""

from matplotlib.cm import get_cmap
from plot_utils import set_size
from scipy.special import comb
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


plt.rcParams['font.size'] = 7                  # Schriftgroesse
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['lines.linewidth'] = 1           # Linienbreite
plt.rcParams['lines.markersize'] = 3
plt.rcParams['figure.figsize'] = (3.5,4)#set_size(252,subplots=(7,4))
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 1.
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 1.
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.pad']=1.5
plt.rcParams['ytick.major.pad']=1.5
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.handlelength'] = plt.rcParams['legend.handleheight']
plt.rcParams['legend.title_fontsize'] = 8
plt.rcParams['legend.columnspacing'] = 0.8
plt.rcParams['legend.handletextpad'] = 0.4

plt.rcParams['axes.linewidth'] = 0.5
max_deg = 7

cmap = get_cmap('CMRmap')
data_color = np.linspace(0.8,0.15,max_deg)
degree_colors = cmap(data_color)


gamma = 60
beta = 0.4
max_deg = 7
test_length = '5e4'

fmt = 'eps'

#path = '../analog/phi_sweep_gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_long/results/test'+test_length+'/threshold/'
path = '../data/capacities/phi_sweep_gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_long/results/threshold/'

fig = plt.figure()
gs = GridSpec(max_deg,2,figure=fig,width_ratios=[6,1],hspace=1,wspace=0.3)
kwargs = {}
axes = []
max_dels = np.genfromtxt('../data/thresholds/maximum_delays.dat')
thresh_idx_total = np.genfromtxt('../data/thresholds/threshold_plot_data.dat')

ax_label = fig.add_subplot(gs[:,0])
ax_label.xaxis.set_visible(False)
plt.setp(ax_label.spines.values(), visible=False)
ax_label.tick_params(left=False, labelleft=False)
ax_label.patch.set_visible(False)
ax_label.set_ylabel('information processing capacity',labelpad=25)

bar_ax = fig.add_subplot(gs[:,1])
bar_ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
bar_ax.set_ylim([0,110])
bar_ax.set_yticks(range(0,100,5),minor=True)
bar_ax.set_yticks(range(0,110,10))
bar_ax.set_yticklabels(range(0,110,10))
for label in bar_ax.axes.get_yticklabels()[1::2]:
    label.set_visible(False)

interim = 0
for deg in range(1,max_deg+1):
    ax = fig.add_subplot(gs[deg-1,0])
    axes.append(ax)
    deg_scores = np.genfromtxt(path+'gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_phi575_deg'+str(deg)+'.dat',invalid_raise=False)

    if deg_scores.size > 0:
        scores = deg_scores[:,-1]
    del deg_scores

    ax.loglog(scores,color=degree_colors[deg-1],rasterized=True)
    del scores
    ax.set_ylim([1e-5,1])
    maxidx = comb(max_dels[deg-1],deg,repetition=True)
    ax.set_xlim([1,maxidx])

    # xticks
    maxidxlog = int(np.log10(maxidx))+1
    major_ticks = np.logspace(0,maxidxlog,num=maxidxlog+1,endpoint=True)
    minor_ticks = np.unique(np.concatenate([np.linspace(major_ticks[i],major_ticks[i+1],10) for i in range(major_ticks.size-1)]))
    ax.set_xticks(major_ticks[1:-1])
    ax.set_xticks(minor_ticks[1:-1],minor=True)

    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=7)

    # yticks
    ax.set_yticks([1e-1,1e-2,1e-3,1e-4],minor=True)
    ax.set_yticks([1e-1,1e-4])
    ax.set_yticklabels([],minor=True)
    ax.get_yaxis().get_major_formatter().labelOnlyBase = False

    threshline = ax.axhline(y=thresh_idx_total[0,deg-1],
                            color='black',rasterized=True,linewidth=1)
    idxline = ax.axvline(x=thresh_idx_total[1,deg-1],color='black',
                         rasterized=True,linewidth=1)

    degree_capacity = thresh_idx_total[2,deg-1]
    bar_ax.bar(0,degree_capacity,bottom=interim,color=degree_colors[deg-1],rasterized=True)
    bar_ax.set_ylim([0,100])
    interim += degree_capacity

axes[-1].set_xlabel('index of basis function',labelpad=0)

# legend
ph = [bar_ax.plot([],marker="", ls="")[0]]
handles = ph + [Patch(color=degree_colors[d]) for d in range(max_deg)]
labels = ['degree:'] + [str(d+1) for d in range(max_deg)]
leg = fig.legend(ncol=max_deg+1,bbox_to_anchor=(0.45, 0.04),loc='upper center',handles=handles,labels=labels)


plt.savefig(fmt+'/fig12_threshold_estimation.'+fmt,dpi=300,bbox_inches='tight')
