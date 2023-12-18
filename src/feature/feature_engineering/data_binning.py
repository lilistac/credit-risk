'''
Binning class for binning numerical features into categorical features
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rank_IV(iv):
    '''Rank IV due to its predictive power'''

    if iv <= 0.02:
        return 'Useless'
    elif iv <= 0.1:
        return 'Weak'
    elif iv <= 0.3:
        return 'Medium'
    elif iv <= 0.5:
        return 'Strong'
    else:
        return 'suspicious'
        
class Binning:
    def __init__(self, feature, target):
        '''
        Parameters
        ----------
        feature : DataFrame
            DataFrame of features
        target : Series
            Series of target
        '''
        
        self.X = feature
        self.y = target
        self.data = feature.copy()
        self.data['TARGET'] = target

        self.iv = {}
        self.bins = {}

    def _bin_table(self, col, nbins=10, bins=None):
        '''
        Create bin table for a feature
        
        Parameters
        ----------
        col : str
            Feature name
        nbins : int, optional
            Number of bins, by default 10
        bins : list, optional
            List of bins, by default None
        '''

        feature = self.data[[col, 'TARGET']].copy()
        feature = feature.sort_values(col)
        
        # Create bins
        if feature[col].dtype == object:
            feature['bins'] = feature[col]
        else:
            if bins:
                feature['bins'] = pd.cut(self.X[col], bins=bins)
            else:
                # Error due to skewness distribution
                error = True
                while error:
                    try:
                        feature['bins'] = pd.qcut(feature[col], q=nbins)
                        error = False
                    except:
                        nbins -= 1 
        
        # Create bin table
        df_summary = pd.pivot_table(feature, index=['bins'], values=['TARGET'],
                                    columns=['TARGET'], aggfunc={'TARGET': np.size},
                                    fill_value=0)

        df_summary.columns = ['No_Bad', 'No_Good']
        df_summary['No_Observation'] = df_summary['No_Bad'] + df_summary['No_Good']
        return df_summary, nbins
    
    def calculate_WOE(self, colname, nbins=7, min_obs=100, bins=None):
        '''
        Calculate weight of evidence for a feature
        
        Parameters
        ----------
        colname : str
            Feature name
        nbins : int, optional
            Number of bins, by default 7
        min_obs : int, optional
            Minimum number of observations in each bin, by default 100
        bins : list, optional
            List of bins, by default None
        '''

        df_summary, nbins = self._bin_table(colname, nbins=nbins, bins=bins)

        # Replace 0 value of No_Bad in df_summary by 1 to avoid division by 0 error
        df_summary['No_Bad'].replace(0, 1, inplace=True)
        coltype = self.X[colname].dtype

        # Exclude bins with small number of observations
        exclude_ind = np.where(df_summary['No_Observation'] <= min_obs)[0]
        while exclude_ind.shape[0] > 0:
            if coltype == object:
                df_summary.sort_values(by='No_Observation', ascending=False, inplace=True)
                df = pd.DataFrame(df_summary.iloc[-2:].sum(axis=0), columns=['Other']).T
                drop_cols = list(df_summary.iloc[-2:].index)
                df_summary.drop(drop_cols, inplace=True)
                df_summary = pd.concat([df_summary, df])
                exclude_ind = np.where(df_summary['No_Observation'] <= min_obs)[0]
            else:
                nbins -= 1 
                df_summary = self._bin_table(colname, nbins=nbins)
                exclude_ind = np.where(df_summary['No_Observation'] <= min_obs)[0]

        df_summary['GOOD/BAD'] = df_summary['No_Good']/df_summary['No_Bad']
        df_summary['%BAD'] = df_summary['No_Bad']/df_summary['No_Bad'].sum()
        df_summary['%GOOD'] = df_summary['No_Good']/df_summary['No_Good'].sum()
        df_summary['WOE'] = np.log(df_summary['%GOOD']/df_summary['%BAD'])
        df_summary['IV'] = (df_summary['%GOOD']-df_summary['%BAD'])*df_summary['WOE']
        df_summary['Feature'] = colname
        IV = df_summary['IV'].sum()
        return df_summary, IV, nbins
    
    def plot_woe(self, col, nbin, rot=0):
        '''Plot weight of evidence by a feature'''

        df_summary = self.calculate_WOE(col, nbins=nbin)
        plt.figure(figsize=(8,4))
        sns.pointplot(x=df_summary.index, y='WOE',data=df_summary, color='orange')
        plt.title(str('Weight of Evidence by ' + df_summary.columns[0]))
        plt.axhline(y=0, color='grey')
        plt.xticks(rotation = rot)
        plt.xlabel(' ')
        
    def find_max_IV(self):
        '''Find the maximum IV for each feature'''

        for col in self.X:
            self.iv[col] = 0 
            self.bins[col] = 1
            for i in range(1, 15):
                iv, nbin = self.calculate_WOE(col, bins=i)[1:]

                if iv > self.iv[col]:
                    self.iv[col] = iv
                    self.bins[col] = nbin

                if nbin < i:
                    break

    def create_IV_info(self, file_path='src/feature/iv.csv', bins=None):
        '''
        Create IV dataframe for all features

        Parameters
        ----------
        file_path : str, optional
            File path to save IV dataframe, by default 'iv.csv'
        bins : dict, optional
            Dictionary of bins for each feature, by default None
        '''

        if not bins:
            self.find_max_IV()
        else:
            self.iv = {}
            for col in self.X:
                print('Finding IV for', col)
                self.iv[col] = self.calculate_WOE(col, bins=bins[col])[1]
        df_IV = pd.DataFrame(self.iv.items(), columns=['Features', 'IV'])
        df_IV['Rank'] = df_IV['IV'].apply(lambda x: rank_IV(x))
        df_IV = df_IV.sort_values('IV', ascending=False).reset_index(drop=True)
        df_IV.to_csv(file_path, index=False)
        return df_IV