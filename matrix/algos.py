import numpy as np
import pandas as pd

import time
import datetime
import math
import pickle

import ffn

from sklearn import linear_model
from numpy.lib.stride_tricks import as_strided

class Algo(object):
    def __init__(self):
        pass

class CalculateMomentum(Algo):
    def __init__(self, window = 120):
        super(CalculateMomentum, self).__init__()
        self.window = window
        
    def __call__(self, states):
        # NOTE: MODIFYING STATES HERE WILL BE MODIFYING ORIGINAL VALUES
        # THIS IS DELIBERATE
        states['score'] = states['data'].pct_change(self.window)
        return states

class CalculateInvVol(Algo):
    def __init__(self, window = 120):
        super(CalculateInvVol, self).__init__()
        self.window = window
    
    def __call__(self, states):
          
        returns = states['data'].pct_change().fillna(method='ffill')
        vol = 1.0 / returns.rolling(self.window).std()
        vol[np.isinf(vol)] = np.NaN
        vol[vol>1000000] = np.NaN # @HACK/TODO: for too small volatility?
        vols = vol.sum(axis=1)
        weights = vol.divide(vols, axis = 0)
        
        states['score'] = weights

        return states

class SelectN(Algo):
    def __init__(self, n, ascending = False):
        super(SelectN, self).__init__()
        self.n = n
        self.ascending = ascending
    
    def __call__(self, states):
        states['selected'] = states['score'].rank(axis=1, ascending=self.ascending) <= self.n
        return states

class SelectNPct(Algo):
    def __init__(self, pct, ascending = False):
        super(SelectNPct, self).__init__()
        self.pct = pct
        self.ascending = ascending
    
    def __call__(self, states):
        states['selected'] = states['score'].rank(axis=1, ascending=self.ascending).le(states['score'].count(axis=1) * self.pct, axis=0)
        return states

class SelectAll(Algo):
    def __init__(self):
        super(SelectAll, self).__init__()
    
    def __call__(self, states):
        states['selected'] = ~states['data'].isna()
        return states

class SelectWhere(Algo):
    def __init__(self, cond):
        super(SelectWhere, self).__init__()
        self.cond = cond
    
    def __call__(self, states):
        states['selected'] = states['data'][self.cond]
        return states

class WeighEqually(Algo):
    def __init__(self):
        super(WeighEqually, self).__init__()
        
    def __call__(self, states):
        
        selected = states['selected']
        data = states['data']
        
        counts = data[selected].count(axis=1)
        counts[counts==0] = np.NaN
        rolling_weights = pd.DataFrame(index=data.index, columns=data.columns)
        rolling_weights[selected] = 1

        rolling_weights = rolling_weights.divide(counts, axis=0)
        
        states['weights'] = rolling_weights
        
        return states

class WeighSpecific(Algo):
    def __init__(self, specific_weights):
        super(WeighSpecific, self).__init__()
        self.specific_weights = specific_weights
        
    def __call__(self, states):
        
        data = states['data']
        
        rolling_weights = pd.DataFrame(index=data.index, columns=data.columns)
    
        for k, w in self.specific_weights.items():
            rolling_weights[k] = w

        states['weights'] = rolling_weights
        
        return states

class Weigh(Algo):
    def __init__(self, weights):
        super(Weigh, self).__init__()
        self.weights = weights
        
    def __call__(self, states):
        
        states['weights'] = self.weights
        
        return states

class WeighInvVol(Algo):
    def __init__(self, window = 120):
        super(WeighInvVol, self).__init__()
        self.window = window
    
    def __call__(self, states):
        
        selected = states['selected']
        
        returns = states['data'].pct_change().fillna(method='ffill')
        vol = 1.0 / returns.rolling(self.window).std()

        # Below lines to make sure that if an asset is not selected (NaN in data),
        # it will be excluded from calculation.
        vol[np.isnan(selected)] = np.NaN 
        vol[selected==False] = np.NaN
        vol[np.isinf(vol)] = np.NaN
        vol[vol>1000000] = np.NaN # @HACK/TODO: for too small volatility?
        vols = vol.sum(axis=1)
        weights = vol.divide(vols, axis = 0)
        
        states['weights'] = weights

        return states

class WeighBetaHedge(Algo):
    def __init__(self, window = 30, major = '', hedge = ''):
        super(WeighBetaHedge, self).__init__()
        self.window = window
        self.major = major
        self.hedge = hedge
    
    def __call__(self, states):
    
        returns = states['data'].pct_change().fillna(method='ffill')
        major = returns[self.major]
        hedge = returns[self.hedge]
        
        beta = major.rolling(self.window).cov(hedge) / hedge.rolling(self.window).var()
        
        rolling_weights = pd.DataFrame(index=returns.index, columns=returns.columns)
        rolling_weights[self.major] = 1
        rolling_weights[self.hedge] = -beta

        states['weights'] = rolling_weights

        return states

class TargetVol(Algo):
    def __init__(self, target = 0.2, window = 250, annualization_factor = 252):
        super(TargetVol, self).__init__()
        self.target = target
        self.window = window
        self.annualization_factor = annualization_factor
    
    def __call__(self, states):

        selected = states['selected']
        data = states['data']
    
        rolling_weights = pd.DataFrame(index=data.index, columns=data.columns)
        rolling_weights[:] = 1

        returns = data.pct_change()
        vol = returns.rolling(self.window).std() * np.sqrt(self.annualization_factor)

        resize = self.target / vol

        resize[np.isinf(resize)] = np.NaN
        resize[resize>10000] = np.NaN # @HACK/TODO: for too small volatility?

        rolling_weights = rolling_weights * resize
        
        rolling_weights[np.isnan(selected)] = np.NaN 
        rolling_weights[selected==False] = np.NaN
        
        states['weights'] = rolling_weights

        return states

class WeightCap(Algo):
    def __init__(self, cap = 0.2):
        super(WeightCap, self).__init__()
        self.cap = cap
        
    def __call__(self, states):
        """
        This only deals with rolling target weights, not the actual weights 
        that might changed due to market returns
        """
        
        w = states['weights'].copy()
        w[w > self.cap] = self.cap
        w[w < -self.cap] = -self.cap
        
        states['weights'] = w
        
        return states

class WeightRounding(Algo):
    def __init__(self, threshold = 0.1):
        super(WeightRounding, self).__init__()
        self.threshold = threshold
        
    def __call__(self, states):
        """
        This only deals with rolling target weights, not the actual weights 
        that might changed due to market returns
        """
        
        w = states['weights'].copy()
        w[w.abs() < self.threshold] = 0
        states['weights'] = w
        
        return states

class CloseAt(Algo):
    def __init__(self, dates, next_reopen = False):
        super(CloseAt, self).__init__()
        self.dates = dates
        self.next_reopen = next_reopen
        
    def __call__(self, states):

        weights = states['weights'].copy()
        
        for date in self.dates:
            date_id = weights.index.get_loc(date)
            for c in weights.columns:
                if self.next_reopen:
                    weights[c].iloc[date_id+1] = weights[c].loc[date]
                weights[c].loc[date] = 0
        
        states['weights'] = weights
        
        return states

class Rebalance(Algo):
    def __init__(self, freq = 'M'):
        super(Rebalance, self).__init__()
        self.freq = freq
        
    def _get_rebalance_dates(self, df, freq='M'):
        if freq == 'M':
            dates1 = pd.Series(df.index.month)
            dates2 = pd.Series(df.index.month).shift(1)
        elif freq == 'Q':
            dates1 = pd.Series(df.index.quarter)
            dates2 = pd.Series(df.index.quarter).shift(1)
        elif freq == 'W':
            dates1 = pd.Series(df.index.week)
            dates2 = pd.Series(df.index.week).shift(1)
        else:
            dates1 = pd.Series(df.index.day)
            dates2 = pd.Series(df.index.day).shift(1)

        is_rebalance = (dates1 != dates2)
        rebalance_dates = list(df.index[is_rebalance])

        return rebalance_dates
    
    def __call__(self, states):
        
        rebalance_dates = self._get_rebalance_dates(states['weights'], freq=self.freq)
        
        weights = pd.DataFrame(index=states['weights'].index, columns=states['weights'].columns)
        weights.loc[rebalance_dates] = states['weights']
        
        states['weights'] = weights
        
        return states