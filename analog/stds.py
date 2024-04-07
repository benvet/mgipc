import numpy as np
gamma_beta = [(20,0.4),(60,0.4),(180,0.4),(60,0.2),(60,0.8)]
phis = np.array([12, 50, 87, 125, 162, 200, 237, 275, 312, 350, 387, 425, 462, 500, 537, 575, 612, 650, 687, 725])

washout = 10000

stds = np.zeros((phis.size,len(gamma_beta)+1))
stds[:,0] = phis

for i,phi in enumerate(phis):
	for j,(gamma, beta) in enumerate(gamma_beta):
		path = '../data/capacities/phi_sweep_gamma'+str(gamma)+'_beta'+str(beta).replace('.','-')+'_long/measurements/'
		filename = 'measoutfilefeed'+str(beta)+'scal'+str(gamma)+'phi'+str(phi)+'long.txt'
		X = np.genfromtxt(path+filename)[washout:]
		stds[i,j+1] = np.std(X)

np.savetxt('../data/stds/stds.dat',stds,header='phi '+' '.join(str(gb) for gb in gamma_beta),fmt='%.5f')
