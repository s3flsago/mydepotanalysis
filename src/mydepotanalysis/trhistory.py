import numpy as np
from numpy.lib.type_check import _asfarray_dispatcher
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
from .params import CURRENCIES, INSTITUTIONS, price_matrix
from .transaction import TR_TYPES, Trade, InOutflow
from copy import copy, deepcopy

class TrHistory(object):
    def __init__(self, trs=[]):
        self.transactions = []
        self.iterno = 0 
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.iterno<len(self.transactions):
            tr = self.transactions[self.iterno]
            self.iterno += 1
            return tr
        else:
            self.iterno = 0
            raise StopIteration
    
    def __len__(self):
        return len(self.transactions)

    def add_Tr(self, tr): # tr should be Transaction object
        self.transactions.append(tr)

    def get_Tr(self, ind): # by index
        return self.transactions[ind]
    
    def reindex(self, indices=None):
        trh = [self].copy()[0]
        if indices==None:
            indices = np.arange(len(self.transactions))
        print(indices)
        for tr, ind in zip(self.transactions, indices):
            tr.set_index(ind)
        return trh
        
    def display(self):
        df = pd.DataFrame(columns=['tr_index', 'tr_type', 'time', 'currency_out', 'currency_in', 'price_out', 'price_in', 'q_out',
                                  'q_in', 'Fee (€)', 'institution_in', 'institution_out', 'institution'])
        for tr in self.transactions:
            if type(tr)==Trade:
                new_row = {'tr_index': tr.index, 'tr_type': 'Trade', 'time': tr.time, 'currency_out': tr.currency_out, 
                           'currency_in': tr.currency_in, 'price_out': tr.price_out, 'price_in': tr.price_in, 
                           'q_out': tr.q_out,
                           'q_in': tr.q_in, 'Fee (€)': tr.fee, 'institution_in': '', 'institution_out': '', 
                           'institution': tr.institution}
            if type(tr)==InOutflow:
                new_row = {'tr_index': tr.index, 'tr_type': 'InOutflow', 'time': tr.time, 'currency_out': tr.currency_out, 
                           'currency_in': tr.currency_in, 'price_out': '', 'price_in': tr.price_in, 
                           'q_out': tr.q_out, 'q_in': tr.q_in, 'Fee (€)': '', 
                           'institution_in': tr.institution_in, 'institution_out': tr.institution_out, 
                           'institution': ''}
            df = df.append(new_row, ignore_index=True)
        df = df.set_index('tr_index')
        display(df)
        return df
         
    def ffilter(self, institutions=INSTITUTIONS, currencies=CURRENCIES, tr_types=TR_TYPES):
        res = TrHistory()
        for tr in iter(self):
            cond1 = len(set([tr.institution_in, tr.institution_out]) & set(institutions))!=0
            cond2 = len(set([tr.currency_in, tr.currency_out]) & set(currencies))!=0
            cond3 = any([isinstance(tr, el) for el in tr_types])
            if cond1 and cond2 and cond3:
                res.add_Tr(copy(tr))
        return res

    def get_purchase_time(self, curr, get_profits=True, show=False):
        newTrH = self.ffilter(currencies=[curr])
        newTrH.curr = curr

        exp = {'t': np.array([]), 'q': np.array([]), 'pEUR': np.array([])}
        profits = {'t': np.array([]), 'q': np.array([]), 'profitEUR': np.array([])}
        for tr in newTrH.transactions:
            if tr.currency_in==curr:
                exp['t'] = np.append(exp['t'], tr.time) 
                exp['q'] = np.append(exp['q'], tr.q_in)
                exp['pEUR'] = np.append(exp['pEUR'], tr.price_in)
            else:
                profit = 0
                q = 0
                while tr.q_out>0:
                    tmp = deepcopy(exp['q'][0])
                    profit += (tr.price_out - exp['pEUR'][0]) * tmp
                    q += tr.q_out
                    exp['q'][0] -= tr.q_out
                    tr.q_out -= tmp
                    if exp['q'][0]<0:
                        exp['t'] = np.delete(exp['t'], 0)
                        exp['q'] = np.delete(exp['q'], 0)
                        exp['pEUR'] = np.delete(exp['pEUR'], 0)
                        profits['profitEUR'] = np.append(profits['profitEUR'], profit)
                        profits['t'] = np.append(profits['t'], tr.time)
                        profits['q'] = np.append(profits['q'], q)
                    elif tr.q_out<=0:
                        profits['profitEUR'] = np.append(profits['profitEUR'], profit)
                        profits['t'] = np.append(profits['t'], tr.time)
                        profits['q'] = np.append(profits['q'], q)
        exp['qEUR'] = price_matrix.loc['EUR', curr] * exp['q']
        exp['dt'] =  datetime.now() - exp['t'] 
        newTrH.ptime = exp
        newTrH.ptime_agg = pd.DataFrame(exp)[['q', 'qEUR']].sum(axis=0).to_dict()
        newTrH.profits_realised = profits
        newTrH.profits_agg = pd.DataFrame(profits)[['q', 'profitEUR']].sum(axis=0).to_dict()
        if show:
            display(pd.DataFrame(newTrH.ptime))
            display(pd.DataFrame(newTrH.ptime_agg, index=[0]))
            display(pd.DataFrame(newTrH.profits_realised))
            display(pd.DataFrame(newTrH.profits_agg, index=[0]))
        return newTrH

    def analyze_agg(self, curr, show=False):
        # does not work for EUR
        newTrH = self.ffilter(currencies=[curr])
        newTrH.curr = curr

        def get_iter_inout(inout):
            def filter_in(el):
                return el.currency_in==curr 
            def filter_out(el):
                return el.currency_out==curr
            if inout=='in':
                return filter(filter_in, newTrH.transactions)
            else:
                return filter(filter_out, newTrH.transactions)

        newTrH.q = np.array([tr.q_in-tr.q_out for tr in newTrH.transactions]).sum()

        newTrH.q = sum([tr.q_in for tr in get_iter_inout('in')])
        newTrH.q -= sum([tr.q_out for tr in get_iter_inout('out')])

        newTrH.fees = np.array([tr.fee for tr in newTrH.transactions]).sum()

        newTrH.init_value_EUR = sum([tr.price_in*tr.q_in for tr in get_iter_inout('in')])
        newTrH.init_value_EUR -= sum([tr.price_out*tr.q_out for tr in get_iter_inout('out')])

        rate = price_matrix.loc['EUR', curr]
        newTrH.value_EUR = sum([tr.q_in*rate for tr in get_iter_inout('in')])
        newTrH.value_EUR -= sum([tr.q_out*rate for tr in get_iter_inout('out')])

        #newTrH.wgt_agv_time = 0
        #newTrH.wgt_avg_init_price_EUR = 0

        newTrH.ttl_profit_EUR = newTrH.value_EUR - newTrH.init_value_EUR
        #newTrH.rtl_ttl_profit = (newTrH.value_EUR / newTrH.init_value_EUR - 1) * 100
        #newTrH.profit_factor = newTrH.value_EUR / newTrH.init_value_EUR
        
        if show: newTrH.display()
        return newTrH
    
    def check_tr_development(self, indices):
        TrH = TrHistory()
        for i in indices:
            TrH.add_Tr(self.get_Tr(i))
        TrH.analyze_agg('BTC')
        TrH.display()
        return TrH
