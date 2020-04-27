import numpy as np
import pandas as pd

import time
import datetime
import math
import pickle

class Node(object):
    
    """
    Basic node.
    
    Args:
    - name (str): The Node name
    
    Attributes:
    - prices: Historical prices

    """
    
    def __init__(self, name):
        self.name = name
        
    @property
    def prices(self):
        raise NotImplementedError()

class Security(Node):
    
    """
    Secueriry Node.
    
    Args:
    - name (str): The Node name
    
    Attributes:
    - prices: Historical prices

    """

    # Requirement of pataframe type
    def __init__(self, name, data, ref_col = 'close'):
        Node.__init__(self, name = name)
        self.data = data
        self.ref_col = ref_col
        
    @property
    def prices(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.ref_col]
        else:
            return self.data

class Strategy(Node):
    
    """
    Strategy Node.
    
    Args:
    - name (str): The Node name
    - nodes (list): List of child nodes
    - algos (list): List of algos
    - base_value (int): Base value of strategy
    
    Attributes:
    - prices: Historical prices
    - weights: Historical weights
    - security_weights: Security weights
    - security_data: Data for underlying securities 
    - security_values: Values for underlying securities
    - security_positions: Historical positions for underlying securities
    - transactions: Transactions for underlying securities

    Methods:
    - calculate(): Calculate the historical prices

    """

    def __init__(self, name, nodes = [], algos = [], base_value = 1000000):
        
        Node.__init__(self, name = name)
        self.nodes = nodes
        self.algos = algos
        self.base_value = base_value
        
        # Mard the newly initialized node as new, so calculation will be trigger
        self._is_prices_updated = False
        self._is_security_weights_updated = False
        self._is_security_data_updated = False
        self._is_security_values_updated = False
        self._is_security_positions_updated = False
        self._is_transactions_updated = False
        
    def calculate(self):
        """
        Calculate price index
        """
        
        # This method will update following:
        self._prices = None
        self._weights = None
        
        # (1) Generate combined dataframe from child nodes
        
        data_dict = {}
        for node in self.nodes:
            data_dict[node.name] = node.prices
        data = pd.DataFrame(data_dict)
        
        data = data.fillna(method='ffill')
        returns = data.pct_change()
        
        self.data = data
        
        # (2) Calculate the target weights for rebalance
        
        # The states dict is used to store all necessary information
        # for computation of the weights for nodes and will be passed
        # along in the algo stacks
        
        states = {
            'data' : data,
            'weights' : pd.DataFrame(index=data.index, columns=data.columns)
        }
        
        for algo in self.algos:
            states = algo(states)
            
        # Storing states for debug purpose
        # self.states = states
            
        target_weights = states['weights']
        target_weights_changed_index = target_weights.dropna(how='all').index
            
        # (3) Rebalance (GROSS INDEX)
        
        # Assuming all capital (base) is initially cash. Uninvested capital
        # will be released back as cash.
        cash = self.base_value
        
        # Value(s) refers to value of capital allocated
        value = 0
        values = pd.Series(index=data.index)
        nodes_value = {} # @TODO: Should we keep a time-series history?
        
        # target_weights is the weights target; weights is the actual weights
        weights = pd.DataFrame(index=data.index, columns=data.columns)
        
        for index, row in data.iterrows():
            
            # Note: assuming all calculations are done on T day EOD
            # when all values are known.
            
            # (i) Update value of existing positions
            for name, value in nodes_value.items():
                nodes_value[name] = nodes_value[name] * (1 + returns.loc[index][name])
            
            # IF THERE IS FUNDING COST/INTEREST, CHARGE TO CASH ACCOUNT HERE (NET PRICE)
            # cash = cash + financing()
                
            # (ii) Rebalance if target weights changed
            if (index in target_weights_changed_index):
                value = sum(nodes_value.values()) + cash
                nodes_value = {}
                for name, tweight in target_weights.loc[index].dropna().items():
                    nodes_value[name] = value * tweight
                cash_weight = 1 - target_weights.loc[index].sum()
                cash = cash_weight * value
            
            # (iii) Update value and weights
            value = sum(nodes_value.values()) + cash
            values.loc[index] = value
            for name, node_value in nodes_value.items():
                weights.loc[index][name] = node_value / value

        self._weights = weights
        self._prices = values / self.base_value
        self._is_prices_updated = True
        
    ## MAKING ALL ACTUAL VALUE PRIVATE!!!
        
    @property
    def prices(self):
        if not self._is_prices_updated:
            self.calculate()
        return self._prices
    
    @property
    def weights(self):
        if not self._is_prices_updated:
            self.calculate()
        return self._weights
    
    @property
    def security_weights(self):
        if not self._is_prices_updated:
            self.calculate()
        
        if not self._is_security_weights_updated:
            self._security_weights = self._weights.copy()
            for node in self.nodes:
                if not isinstance(node, Security):
                    node_udl_weights = node.security_weights.multiply(self.weights[node.name], axis=0)
                    self._security_weights = self._security_weights.drop(columns=node.name)
                    self._security_weights = self._security_weights.add(node_udl_weights, fill_value = 0)

        self._is_security_weights_updated = True
        
        return self._security_weights
        
    @property
    def security_data(self):
        if not self._is_security_data_updated:
            self._security_data = self.data.copy()
            for node in self.nodes:
                if not isinstance(node, Security):
                    node_udl_data = node.security_data
                    self._security_data = self._security_data.drop(columns=node.name)
                    self._security_data = pd.concat([self._security_data, node_udl_data], axis=1)
        
        self._security_data = self._security_data.loc[:,~self._security_data.columns.duplicated()]
       
        self._is_security_data_updated = True
        
        return self._security_data
        
    @property
    def security_values(self):
        if not self._is_security_values_updated:
            strategy_value = self.base_value * self.prices
            self._security_values = self.security_weights.multiply(strategy_value, axis=0)
        
        self._is_security_values_updated = True
        
        return self._security_values
    
    @property
    def security_positions(self):
        if not self._is_security_positions_updated:
            self._security_positions = self.security_values / self.security_data
        
        self._is_security_positions_updated = True
        
        return self._security_positions
    
    @property
    def transactions(self, rounding = 2):
        if not self._is_transactions_updated:
            self._transactions = self.security_positions.astype(float).round(rounding).diff()
            self._transactions = self._transactions[self._transactions != 0]
            self._transactions = self._transactions.transpose().unstack().dropna()
        
        self._is_transactions_updated = True
        
        return self._transactions