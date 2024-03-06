import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error

"""
This module contains functions used to simulate an RC setup based on delayed feedback using Mackey-Glass nonlinear virtual nodes.
"""


def nrmse(y_hat,y_t):
    """ Computes the normalized root mean squared error between two sequences.
    """
    y_hat_norm = zscore(y_hat)
    y_t_norm = zscore(y_t)
    #y_hat_norm = y_hat
    #y_t_norm = y_t
    e = np.sqrt(mean_squared_error(y_t_norm,y_hat_norm))
    return e


def mackey_glass_nl(x,c=2.189,b=0.001839,p=10.37):
    """
    This function returns the Mackey-Glass nonlinearity used throughout.
    """
    return c*x/(1+(b**p)*(x**p))

def run_single(input=None,
               n=100,
               beta=0.4,
               gamma=60,
               phi=575,
               seed=None,
               init=None,
               w_out=None,
               b_out=0,
               w_in=None,
               config=None,
               return_mask=False,
               normalize=False,
               discrete=False):
    """
    A single run simulating a delay based RC setup with ring topology in open- or closed-loop operation.

    Parameters:
    ----------
    input: int or np.array
        Input sequence fed to the Mackey-Glass system or - if scalar - number of steps to be used in closed loop operation.
    n: int
        Number of virtual nodes.
    beta: float
        Feedback scaling parameter.
    gamma: float
        Input scaling parameter.
    phi: float
        Bias parameter.
    seed:
        Random seed.
    init: np.array
        Initial state
    w_out: np.array
        Output weights.
    b_out: float
        Output bias
    w_in: np.array
        Input weights.
    config=None,
        Optional config dictionary to be used.
    return_mask: bool
        If True the input mask (=input weights) are returned.
    normalize: bool
        If True, the input is normalized by a zscore transformation.
    discrete: bool
        If True, the system state is rounded to integer values after every update.
    """
    if config:
        n = int(config['n'])
        beta = float(config['beta'])
        gamma = float(config['gamma'])
        phi = float(config['phi'])
        seed = int(config['seed'])

    if seed:
        np.random.seed(seed)

    if np.isscalar(input):
        # closed loop
        steps = input

        if w_out is None:
            print("Error: Can not operate in closed loop without output weights provided.")
            return
        if w_out.shape[0] != n:
            print('Error: output weight dimensions do not match number of nodes.')
            return

        R = np.zeros((steps,n))
        if init is None:
            r = 0
        else:
            r = init[-1]
            R[-2:] = init

        for n in range(steps):
            if n<2:
                r_ = 0
            else:
                r_ = np.concatenate([R[n-2,-1].reshape(1,-1),R[n-1,:-1].reshape(1,-1)],axis=1)

            u = np.dot(r,w_out).reshape(1,-1) + b_out
            x = u*w_in + phi + beta * r_

            r = mackey_glass_nl(x)
            if discrete:
                r = np.round(r,0)
            R[n] = r
    else:
        # open loop
        steps = input.shape[0]
        if normalize:
            U = zscore(input)
        else:
            U = input

        if w_in is None:
            w_in = np.random.normal(scale=gamma,size=(input.shape[-1],n))
        #print('Open loop simulation for T='+str(steps)+' steps. n='+str(n))

        R = np.zeros((steps,n))
        for n in range(steps):
            u = U[n]
            if n < 2:
                # initialization
                r_ = 0
            else:
                # run
                r_ = np.concatenate([R[n-2,-1].reshape(1,-1),R[n-1,:-1].reshape(1,-1)],axis=1)

            x = np.clip(u*w_in + phi + beta * r_,0,1023)

            r = mackey_glass_nl(x)
            if discrete:
                r = np.round(r,0)
            R[n] = r
    if return_mask:
        return R,w_in
    else:
        return R

# Main function just for simple testing purposes
def main():
    # input
    T = 1000
    p = 17
    U = np.sin(2*np.pi*np.linspace(0,T-1,T)/p).reshape(-1,1)

    R = run_single(U)

    plt.plot(R)
    #plt.plot(U*200)
    plt.show()

if __name__=='__main__':
    main()
