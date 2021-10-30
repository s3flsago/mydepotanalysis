from pandas.core import indexing
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt

class Transaction(object):
    def __init__(self, time, institution_in, institution_out, comment='', index=0):
        self.time = time
        self.comment = comment
        self.institution_in = institution_in
        self.institution_out = institution_out
        self.index = index
    
    def set_index(self, ind):
        self.index = ind
    
class InOutflow(Transaction):
    def __init__(self):
        pass
    
    def __init__(self, time=datetime.now(), q_in=1, institution_in='BSDEX', institution_out='ING_GIRO', currency_in='EUR', price_in=1, comment='', index=0):
        self.time = time
        self.q_in = q_in
        self.q_out = q_in * (-1)
        self.currency_in = currency_in
        self.currency_out = currency_in
        self.price_in = price_in
        self.price_out = price_in
        self.institution_in = institution_in
        self.institution_out = institution_out
        self.index = index
    
    def __type__(self):
        return 'InOutflow'
        
class Trade(Transaction):
    def __init__(self, time=datetime.now(), currency_in='BTC', currency_out='EUR', price_in=50000, price_out=1,
                 q_in=1, q_out=50000, fee=1, institution='BSDEX', comment='', index=0):
        self.time = time
        self.currency_in = currency_in
        self.currency_out = currency_out
        self.price_in = price_in
        self.price_out = price_out
        self.q_in = q_in
        self.q_out = q_out
        self.fee = fee
        self.institution = institution
        self.institution_in, self.institution_out = institution, institution
        self.comment = comment
        self.index = index

TR_TYPES = [type(Trade()), type(InOutflow())]
