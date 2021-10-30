import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
from .params import CURRENCIES, INSTITUTIONS, price_matrix
from .transaction import TR_TYPES, Trade, InOutflow
from .trhistory import TrHistory

class Depot(object):
    pass

class StockDepot(Depot):
    pass

class CryptoDepot(Depot):
    def __init__(self, currencies, qs=None):
        self.TransactionHistory = TrHistory()
        self.currencies = currencies
        if qs==None:
            qs = np.zeros(len(currencies))
        self.qs = qs
    
    def __add_(self, depot):
        pass
        
    def trade(self, time, currency_in, currency_out, price_in, price_out, q_in, q_out, fee, institution):
        self.qs[self.currencies.index(currency_out)] -= q_out
        self.qs[self.currencies.index(currency_in)] += q_in
        index = len(self.TransactionHistory)
        tr = Trade(time, currency_in, currency_out, price_in, price_out, q_in, q_out, fee, institution, index=index)
        self.TransactionHistory.add_Tr(tr)
    
    def inoutflow(self, time, currency_in, q_in, institution_in, institution_out):
        self.qs[self.currencies.index(currency_in)] += q_in
        index = len(self.TransactionHistory)
        inout = InOutflow(time, q_in, institution_in, institution_out, currency_in, index=index)
        self.TransactionHistory.add_Tr(inout)
    
    def show_balance(self):
        balance = pd.DataFrame({'currency': self.currencies, 'quantity': self.qs})
        balance['price (€)'] = balance['currency'].apply(lambda x: price_matrix.loc['EUR', x]).round(2)
        balance['value (€)'] = balance['price (€)'] * balance['quantity']
        return balance.set_index('currency')
    
    def show_TrH(self, institutions=INSTITUTIONS, currencies=CURRENCIES, tr_types=TR_TYPES):
        trh = self.TransactionHistory.ffilter(institutions, currencies, tr_types)
        trh.display()

    def read_transactions(self, filename, sheet="Tabelle1", rows=True):
        TAs = pd.read_excel(filename, sheet_name=sheet)
        for index, row in TAs.iterrows():
            if not rows: 
                if index not in np.arange(rows[0], rows[1]): continue
            if row.tr_type=='Trade':
                self.trade(row.time, row.currency_in, row.currency_out, float(row.price_in), float(row.price_out), 
                           float(row.q_in), float(row.q_out), row['Fee (€)'], row.institution_in)
            if row.tr_type=='InOutflow':
                self.inoutflow(row.time, row.currency_in, float(row.q_in), row.institution_in, row.institution_out)
        return TAs
    
    def clear(self):
        self.TransactionHistory = []
        self.qs = np.zeros(len(self.currencies))

    def trhistory_currs(self, currs=CURRENCIES, show=True, transformEUR=True):
        trhs = []
        i = 0
        for curr in currs:
            print(i)
            i +=1
            headline = curr
            trh = self.TransactionHistory.ffilter(INSTITUTIONS, [curr], TR_TYPES)
            if transformEUR and curr=='EUR':
                headline = 'EUR (only InOutFlow)'
                trh = trh.ffilter(INSTITUTIONS, [curr], [type(InOutflow())])
            trhs.append(trh)
            if show:
                print(headline)
                trh.display()
        return trhs
        
    def analyze_agg(self, show=False):
        TrHs = []
        currs = np.array(self.currencies)
        
        for curr in currs[currs!='EUR']:
            TrH = self.TransactionHistory.analyze_agg(curr)
            TrHs.append(TrH)

        self.fees = sum([TrH.fees for TrH in TrHs])
        self.init_value_EUR = sum([TrH.init_value_EUR for TrH in TrHs])
        self.value_EUR = sum([TrH.value_EUR for TrH in TrHs])
        self.ttl_profit_EUR = sum([TrH.ttl_profit_EUR for TrH in TrHs])

        if show:
            for ind, curr in enumerate(currs[currs!='EUR']):
                print(curr)
                df = pd.DataFrame({'Fees': TrHs[ind].fees, 'Investment': TrHs[ind].init_value_EUR, 
                               'Current Value': TrHs[ind].value_EUR, 'Profit EUR': TrHs[ind].ttl_profit_EUR, 
                               'Profit %': TrHs[ind].ttl_profit_EUR*100/TrHs[ind].init_value_EUR}, index=[0]).round(2)
                display(df)
            print('Total:')
            df = pd.DataFrame({'Fees': self.fees, 'Investment': self.init_value_EUR, 
                               'Current Value': self.value_EUR, 'Profit EUR': self.ttl_profit_EUR, 
                               'Profit %': self.ttl_profit_EUR*100/self.init_value_EUR}, index=[0]).round(2)
            display(df)

    def get_AssetAge(self, show=False):
        '''Not tested yet!'''
        TrHs = {}
        AssetAges = None
        for curr in CURRENCIES:
            TrH = Flos_Depot.TransactionHistory.get_purchase_time(c, show=False)
            TrHs[curr] = TrH
        
        pass

   