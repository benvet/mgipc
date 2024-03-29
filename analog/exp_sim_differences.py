import numpy as np
import pandas as pd

gamma_beta = [(20,0.4),(180,0.4),(60,0.2),(60,0.4),(60,0.8)]

phis = np.array([12, 50, 87, 125, 162, 200, 237, 275, 312, 350, 387, 425, 462, 500, 537, 575, 612, 650, 687, 725])
sigmas = np.zeros((36),dtype=int)
sigmas[:20] = np.arange(0,20,dtype=int)
sigmas[20:] = np.arange(20,100,5,dtype=int)
max_deg = 7
shift = 1
test_length = '5e4'
safety_factor = 2
thresholds = np.genfromtxt('../data/thresholds/thresholds_test'+test_length+'.dat') * safety_factor

differences = np.zeros((sigmas.size,max_deg))
degree_score_differences = np.zeros((sigmas.size,max_deg))
for i,(gamma, beta) in enumerate(gamma_beta):
    for j,phi in enumerate(phis[shift:]):
        path = '../data/capacities/phi_sweep_gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_long/'

        for k,sigma in enumerate(sigmas):
            for d in range(1,max_deg+1):
                names = list(range(d)) + ['c']
                #print(names)
                phi_sim = phis[j]
                filename_exp = 'gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_phi'+str(phi)+'_deg'+str(d)+'.dat'

                deg_caps_exp = pd.read_csv(path+'results/'+filename_exp,delimiter=' ',names=names)
                deg_caps_exp_thresh = deg_caps_exp.loc[deg_caps_exp['c']>=thresholds[d-1]]

                filename_sim = 'sigma'+str(sigma)+'_gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_phi'+str(phi_sim) + '_deg'+str(d)+'.dat'
                deg_caps_sim = pd.read_csv(path+'corresponding_simulation/noise/'+filename_sim,delimiter=' ',names=names)
                deg_caps_sim_thresh = deg_caps_sim.loc[deg_caps_sim['c']>=thresholds[d-1]]

                degree_differences = deg_caps_sim_thresh['c'].subtract(deg_caps_exp_thresh['c'],fill_value=0,level=list(range(d)))

                if degree_differences.count() > 0:
                    deg_diff_total_norm =  degree_differences.abs().sum()
                else:
                    deg_diff_total_norm = 0
                degree_score_differences[k,d-1] = np.abs(deg_caps_sim_thresh['c'].sum()-deg_caps_exp_thresh['c'].sum())
                differences[k,d-1] += deg_diff_total_norm

np.savetxt('analog/exp_sim_differences_shift'+str(shift)+'_safety'+str(safety_factor)+'.dat',differences)
