import numpy as np
import matplotlib.pyplot as plt
from scipy import ones_like
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from mgipc.delay_rc import run_single
from scipy.stats import zscore,chi2
from itertools import chain
from os import linesep


def cap(X_train,X_test,y_train,y_test,reg=0.001):
    """ Evaluates a single capacity from the given train/test sets using a Tikhonov regularization factor."""
    lr = Ridge(alpha=reg).fit(X_train,y_train)
    y_hat = lr.predict(X_test)
    return 1 - mse(y_test,y_hat)/np.mean(y_test**2)


class LegendreBasis:
    """ Class representing the basis of Legendre polynomials for a given maximum degree and delay.

    Given an input sequence U, an internal cache is filled with the specified polynomial transformations of the input sequence.
    The input sequence tranformed by a spcific element of the Hilbert space of fading memory functions can then be retrieved 
    by calling get_target with the corresponding index.

    Attributes
    ----------
    max_deg : int
        Maximum degree of basis functions to be computed
    max_del : int
        Maximum delay of variables entering the basis functions.
    train_steps : int
        Determines the number of data points of the input U used for training.
    test_steps : int
        Determines the number of data points of the input U used for testing.
    cache: np.array
        Holds the transformations of the input sequence in ascending order of degree.

    Methods
    -------
    get_target(self,idx)
        Returns the input sequence tranformed by an element of the Hilbert space of fading memory functions as indicated by the corresponding index.
.
    """
    def __init__(self,max_deg,max_del,U,train_steps,test_steps):
        self.max_deg = max_deg
        self.max_del = max_del
        self.train_steps = train_steps
        self.test_steps = test_steps

        # initialize cache
        self.cache = np.zeros((max_deg,U.shape[0],U.shape[1]))
        self.cache[0] = U

        pnm1, pnm2 = U, np.ones_like(U)
        for n in range(2,max_deg+1):
            pn = ((2 * n - 1) * U * pnm1 - (n - 1) * pnm2) / n
            self.cache[n-1] = pn
            pnm2 = pnm1
            pnm1 = pn

    def get_target(self,idx):
        """Returns the input sequence transformed by an element of the Hilbert space of fading memory functions as indicated by the corresponding index.
            A train/test split is applied in accordance with the specified train_steps/test_steps.        
        """
        positions, powers = np.unique(idx,return_counts=True)
        y_train, y_test = np.ones((self.train_steps,1)), np.ones((self.test_steps,1))
        for pos,pow in zip(positions,powers):
            y_train *= zscore(self.cache[pow-1, self.max_del-pos : self.train_steps + self.max_del - pos])
            y_test *= zscore(self.cache[pow-1, self.train_steps + self.max_del-pos : self.train_steps + self.test_steps + self.max_del - pos])
        return y_train, y_test


def heuristic_gen(X_train,
                  X_test,
                  deg,
                  idxs=[],
                  eps=0.0001,
                  base=None,
                  max_del=100,
                  delay=None,
                  patience=10,
                  reg=0.001,
                  heuristic=True):
    """
    Implements a heuristic search for nonzero capacities of a single degree as layed out in the supplementary material of Dambre et al.(2012).
    
    This recursive generator passes indices down to a call of itself with a lower degree until degree 0.
    It then yields the nonlinear memory capacity. If the values fall below the predefined threshold eps the generator terminates.
    
    Parameters:
    ----------
    eps: float
        Threshold below which capacities will be considered zero.
    base: LegendreBasis
        Basis used to construct target functions.
    max_del: int
        Maximum delay used in capacity evaluations.
    patience: int
        Determines how many subsequent capacities will be evaluated after a single capacity evaluated to zero.
    reg: float
        Tikhonov regularization factor
    heuristic: bool
        If False a complete (non-heuristic) capacity search will be performed for the given range of delays.
    """
    if delay is None:
        delay = max_del
    if deg == 0:
        y_train, y_test = base.get_target(idxs)
        score = cap(X_train,X_test,y_train,y_test,reg=reg)
        if heuristic:
            if score >= eps:
                yield idxs, score
            else:
                return 1
        else:
            yield idxs, np.clip(score,0,None)

        #if score >= eps:
        #    yield idxs,score    # yield significant score
        #elif heuristic:
        #    return 1            # terminate due to subthreshold score
    else:
        count_down = patience
        for i in range(delay+1):
            next_idxs = [i] + idxs
            subscore = yield from heuristic_gen(X_train,
                                                X_test,
                                                deg-1,
                                                idxs=next_idxs,
                                                eps=eps,
                                                base=base,
                                                max_del=max_del,
                                                delay=i,
                                                patience=patience,
                                                heuristic=heuristic)
            if subscore: # subgenerator returned because of no more scores
                if count_down:
                    count_down -= 1
                else:
                    return 1
            else:
                count_down = patience


def collect_capacities(X_train,
                       X_test,
                       U,
                       min_deg=1,
                       max_deg=3,
                       max_del=10000,
                       eps=0,
                       patience=10,
                       reg=0.001,
                       heuristic=True,
                       filename='capacities.dat',
                       return_degrees=False):
    """ Collects capacities for basis function degrees in the range (min_deg,max_deg) and stores them under the filename specified.

        Parameters:
        ----------
        X_train: np.array
            System states used for training.
        X_test: np.array
            System states used for testing.
        U: np.array
            Input used to drive the system and evaluate capacities.
        min_deg: int
            Minimum basis function degree.
        max_deg : int
            Maximmum basis function degree.
        max_del: int
            Maximum delay.
        eps=0 : float or 1d array
            Threshold used in capacity evaluations for all degrees or array of degree specific thresholds.
        patience: int
            Patience used in capacity evaluations.
        reg: float
            Tikhonov regularization factor used in capacity evaluations.
        heuristic: bool
            Determines wether heuristic or full capacity search is employed.
        filename: str
            Determines where the found capacities will be stored. 
            If no filename is given, only the total capacity and (if specified) the scores per degree are returned.
        return_degrees: bool
            If True scores per degree are returned. 

    """
        lbase = LegendreBasis(max_deg,max_del.max(),U,X_train.shape[0],X_test.shape[0])

        total_score = 0
        degree_scores = []
        for deg in range(min_deg,max_deg+1):
            deg_score = 0
            if np.isscalar(eps):
                threshold = eps
            else:
                threshold = eps[deg-1]
            if np.isscalar(max_del):
                md = max_del
            else:
                md = max_del[deg-1]
            if filename:
                # record all collected capacities
                deg_filename = filename.split(sep='.')[0] + '_deg'+str(deg) + '.' + filename.split(sep='.')[1]
                with open(deg_filename,'w') as file:
                    for idx,score in heuristic_gen(X_train,
                                                   X_test,
                                                   deg,
                                                   eps=threshold,
                                                   base=lbase,
                                                   max_del=md,
                                                   patience=patience,
                                                   reg=reg,
                                                   heuristic=heuristic):
                        deg_score += score
                        for i in idx:
                            file.write(str(i)+' ')
                        file.write('{:.5f}'.format(score)+linesep)
            else:
                # record only total degree score
                for idx,score in heuristic_gen(X_train,
                                               X_test,
                                               deg,
                                               eps=threshold,
                                               base=lbase,
                                               max_del=md,
                                               patience=patience,
                                               reg=reg,
                                               heuristic=heuristic):
                    deg_score += score
            degree_scores.append(deg_score)
            total_score += deg_score
        if return_degrees:
            return total_score, degree_scores
        else:
            return total_score



# TESTS

def test_collect():
    train_steps = 10000
    test_steps = 2000
    max_del = 1000
    washout = 1000
    steps = train_steps + test_steps + washout + max_del
    min_deg = 1
    max_deg = 3
    p_falsepos = 0.0001
    threshold=2*chi2.isf(1-p_falsepos,100)/train_steps
    print(threshold)
    noise_scale = 0
    np.random.seed(8562154)
    U = np.random.uniform(-1,1,size=(steps,1))

    X = zscore(run_single(input=U)[washout:])
    X += np.random.normal(scale=noise_scale,size=X.shape)

    X_train = X[max_del:max_del+train_steps]
    X_test = X[train_steps + max_del:]



    c_total = collect_capacities(X_train,
                                 X_test,
                                 U[washout:],
                                 min_deg=min_deg,
                                 max_deg=max_deg,
                                 max_del=max_del,
                                 eps=threshold)


if __name__=='__main__':
    test_collect()
