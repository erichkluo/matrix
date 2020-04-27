import numpy as np
import pandas as pd

class Simulation(object):
    
    def __init__(self, positions = None, value_prices = None, trade_prices = None,
                 base_value = 1000000, comms = 0):
        
        self.positions = positions
        self.value_prices = value_prices
        self.trade_prices = trade_prices
        self.base_value = base_value
        self.comms = comms
        self._has_run = False
        
    def run(self):
        
        positions = self.positions.astype(float).fillna(0)
        
        transactions = positions.diff()
        transactions[transactions.abs()<0.01]= np.NaN
        
        value_prices = self.value_prices
        trade_prices = self.trade_prices

        #actual_price = Portfolio(transactions.columns, type=type).cbd_swap['open'].loc[positions.index].shift(-1)
        #value_price = Portfolio(transactions.columns, type=type).cbd_swap['close'].loc[positions.index]

        cash_chg = transactions * trade_prices * -1 - abs(transactions * trade_prices * self.comms)
        cash = cash_chg.sum(axis=1).cumsum() + self.base_value

        self.slippage = ((value_prices - trade_prices) * transactions).sum(axis=1).cumsum()

        #asset_value = (positions * value_prices).sum(axis=1)
        asset_value = (positions * trade_prices).sum(axis=1)

        self._nav = cash + asset_value
        self._has_run = True
        
    @property
    def nav(self):
        if not self._has_run:
            self.run()
        return self._nav