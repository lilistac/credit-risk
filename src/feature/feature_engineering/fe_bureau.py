import pandas as pd
import numpy as np
import time

class Bureau:
    def __init__(self, file_path, index_col=None, verbose=True, bureau_balance_filepath=None):
        '''
        Parameters
        ----------
        file_path : str
            File path of bureau table
        index_col : int, optional
            Index column, by default None
        verbose : bool, optional
            Whether to print out the progress, by default True
        bureau_balance_filepath : str, optional
            File path of bureau_balance table, by default None
        '''

        self.file_path = file_path
        self.index_col = index_col
        self.verbose = verbose
        self.bureau_balance_filepath = bureau_balance_filepath
    
    def load_data(self):
        start_time = time.time()
        self.bureau = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()
        # Create dataframe for aggregation
        self.agg_bureau = pd.DataFrame(self.bureau.SK_ID_CURR.unique(), columns=['SK_ID_CURR'])
        self.agg_bureau.sort_values('SK_ID_CURR', inplace=True)

        # Counts of a client's previous loans
        # Higher number of previous loans -> higher chance of default
        cnt_previous_loan = self.bureau.groupby('SK_ID_CURR')[['SK_ID_BUREAU']].count()
        cnt_previous_loan.rename(columns={'SK_ID_BUREAU': 'CNT_PREV_LOANS'}, inplace=True)
        self.agg_bureau = pd.merge(self.agg_bureau, cnt_previous_loan, on='SK_ID_CURR', how='left')

        # Number of bad debts of each person
        # Higher number of bad debts -> higher chance of default
        bad_dept = self.bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack()[['Bad debt']]
        bad_dept.fillna(0, inplace=True)
        self.agg_bureau = pd.merge(self.agg_bureau, bad_dept, on='SK_ID_CURR', how='left')
        self.agg_bureau.rename(columns={'Bad debt': 'CNT_BAD_DEBT'}, inplace=True)

        # Mode of currency and credit type
        mode_currency = self.bureau.groupby('SK_ID_CURR')[['CREDIT_CURRENCY', 'CREDIT_TYPE']].agg(lambda x: x.mode()[0])
        mode_currency.columns = [col + '_MODE' for col in mode_currency.columns]
        self.agg_bureau = pd.merge(self.agg_bureau, mode_currency, on='SK_ID_CURR', how='left')

        # The gap between 2 latest credit dates and credit update dates
        # The larger the gap, the higher the chance of default
        last_app = self.bureau.sort_values('DAYS_CREDIT').groupby('SK_ID_CURR')[['DAYS_CREDIT', 'DAYS_CREDIT_UPDATE']].last().reset_index()
        last_app.columns = ['SK_ID_CURR', 'LAST_DAYS_CREDIT', 'LAST_DAYS_CREDIT_UPDATE']
        self.agg_bureau = pd.merge(self.agg_bureau, last_app, on='SK_ID_CURR', how='left')
        self.agg_bureau.rename(columns={'DAYS_CREDIT': 'LAST_DAYS_CREDIT'}, inplace=True)

        # The gap between the last credit closed and the current application
        last_end = self.bureau[self.bureau.CREDIT_ACTIVE == 'Closed'].sort_values('DAYS_ENDDATE_FACT').groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].last().reset_index()
        self.agg_bureau = pd.merge(self.agg_bureau, last_end, on='SK_ID_CURR', how='left')
        self.agg_bureau.rename(columns={'DAYS_ENDDATE_FACT': 'LAST_DAYS_ENDDATE_FACT'}, inplace=True)

        # Aggregations of largest overdue debts
        agg = self.bureau.groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM',
                                            'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_MAX_OVERDUE', 
                                            'AMT_ANNUITY']].agg(['min', 'max', 'median', 'sum'])
        agg.columns= [col[0] + '_' + col[1].upper() for col in agg]
        self.agg_bureau = pd.merge(self.agg_bureau, agg, on='SK_ID_CURR', how='left')

        # Timely payments made on or before the credit end date contribute to a positive credit history
        self.bureau['ENDDATE_DIF'] = self.bureau['DAYS_CREDIT_ENDDATE'] - self.bureau['DAYS_ENDDATE_FACT']
        agg = self.bureau.groupby('SK_ID_CURR')[['ENDDATE_DIF']].agg(['min', 'max', 'median', 'sum'])
        agg.columns= [col[0] + '_' + col[1].upper() for col in agg]
        self.agg_bureau = pd.merge(self.agg_bureau, agg, on='SK_ID_CURR', how='left')


        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def add_prefix(self):
        self.agg_bureau.columns = ['BUREAU_' + col for col in self.agg_bureau.columns]
        self.agg_bureau.rename(columns={'BUREAU_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)

    def merge_bureau_balance(self):
        '''Merge bureau and bureau_balance tables'''

        if self.bureau_balance_filepath:
            bureau_balance = BureauBalance(self.bureau_balance_filepath)
            bureau_balance = bureau_balance.execute()
            bureau_merged = pd.merge(self.bureau, bureau_balance, on='SK_ID_BUREAU', how='left')

            start_time = time.time()
            # New features
            # Longer credit durations typically expose to a higher risk of default
            bureau_merged['CREDIT_DURATION'] = np.abs(bureau_merged['DAYS_CREDIT'] - bureau_merged['DAYS_CREDIT_ENDDATE'])

            # Overdue debts are more likely to default
            bureau_merged['FLAG_OVERDUE_RECENT'] = [0 if ele == 0 else 1 for ele in bureau_merged['CREDIT_DAY_OVERDUE']]

            # Higher maximum overdue debts are more likely to default
            bureau_merged['MAX_AMT_OVERDUE_DURATION_RATIO'] = bureau_merged['AMT_CREDIT_MAX_OVERDUE'] / bureau_merged['CREDIT_DURATION']

            # Higher current overdue debts are more likely to default
            bureau_merged['CURRENT_AMT_OVERDUE_DURATION_RATIO'] = bureau_merged['AMT_CREDIT_SUM_OVERDUE'] / bureau_merged['CREDIT_DURATION']

            # High ratio increase the risk of default if the overdue amount is not paid in time.
            bureau_merged['AMT_OVERDUE_DURATION_LEFT_RATIO'] = bureau_merged['AMT_CREDIT_SUM_OVERDUE'] / (bureau_merged['DAYS_CREDIT_ENDDATE'] + 0.00001)

            # Higher number of prolonged debts and max overdue debts are more likely to default
            bureau_merged['CNT_PROLONGED_MAX_OVERDUE_MUL'] = bureau_merged['CNT_CREDIT_PROLONG'] * bureau_merged['AMT_CREDIT_MAX_OVERDUE']

            # Consistent delays in debt repayment may raise concerns about the borrower's ability to fulfill financial obligations
            bureau_merged['CNT_PROLONGED_DURATION_RATIO'] = bureau_merged['CNT_CREDIT_PROLONG'] / (bureau_merged['CREDIT_DURATION'] + 0.00001)

            # Higher current debt to credit ratio may increase the risk of default
            bureau_merged['CURRENT_DEBT_TO_CREDIT_RATIO'] = bureau_merged['AMT_CREDIT_SUM_DEBT'] / (bureau_merged['AMT_CREDIT_SUM'] + 0.00001)

            # Higher difference between current credit and debt may decrease the risk of default
            bureau_merged['CURRENT_CREDIT_DEBT_DIFF'] = bureau_merged['AMT_CREDIT_SUM'] - bureau_merged['AMT_CREDIT_SUM_DEBT']
            
            # A higher ratio implies a larger portion of the borrower's income allocated to servicing the debt
            bureau_merged['AMT_ANNUITY_CREDIT_RATIO'] = bureau_merged['AMT_ANNUITY'] / (bureau_merged['AMT_CREDIT_SUM'] + 0.00001)

            # The latest Credit Bureau information close to loan application indicates recent and accurate credit behavior data
            bureau_merged['CREDIT_ENDDATE_UPDATE_DIFF'] = np.abs(bureau_merged['DAYS_CREDIT_UPDATE'] - bureau_merged['DAYS_CREDIT_ENDDATE'])
            
            # Aggregations
            aggregations = {
                    'DAYS_CREDIT' : ['mean','min','max','last'],
                    'CREDIT_DAY_OVERDUE' : ['mean','max'],
                    'DAYS_CREDIT_ENDDATE' : ['mean','max'],
                    'DAYS_ENDDATE_FACT' : ['mean','min'],
                    'DAYS_CREDIT_UPDATE' : ['mean','min'],
                    'CREDIT_DURATION' : ['max','mean'],
                    'FLAG_OVERDUE_RECENT': ['sum'],
                    'MAX_AMT_OVERDUE_DURATION_RATIO' : ['max','sum'],
                    'CURRENT_AMT_OVERDUE_DURATION_RATIO' : ['max','sum'],
                    'AMT_OVERDUE_DURATION_LEFT_RATIO' : ['max', 'mean'],
                    'CNT_PROLONGED_MAX_OVERDUE_MUL' : ['mean','max'],
                    'CNT_PROLONGED_DURATION_RATIO' : ['mean', 'max'],
                    'CURRENT_DEBT_TO_CREDIT_RATIO' : ['mean', 'min'],
                    'CURRENT_CREDIT_DEBT_DIFF' : ['mean','min'],
                    'AMT_ANNUITY_CREDIT_RATIO' : ['mean','max','min'],
                    'CREDIT_ENDDATE_UPDATE_DIFF' : ['max','min'],
                    'STATUS_MEAN' : ['mean', 'max'],
                    'WEIGHTED_STATUS_MEAN' : ['mean', 'max']
                    }

            agg = bureau_merged.groupby('SK_ID_CURR').agg(aggregations)
            agg.columns = [col[0] + '_' + col[1].upper() for col in agg.columns]
            self.agg_bureau = pd.merge(self.agg_bureau, agg, on='SK_ID_CURR', how='left')

            if self.verbose:
                print(f'Merging new features bureau and bureau_balance')
                print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
                print('-'*50)

    def execute(self):
        '''Execute the feature engineering pipeline'''

        self.load_data()
        self.feature_engineering()
        self.add_prefix()
        self.merge_bureau_balance()
        return self.agg_bureau
    
class BureauBalance:
    def __init__(self, file_path, index_col=None, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of bureau_balance table
        index_col : int, optional
            Index column, by default None
        verbose : bool, optional
            Whether to print out the progress, by default True    
        '''

        self.file_path = file_path
        self.index_col = index_col
        self.verbose = verbose

    def load_data(self):
        start_time = time.time()
        self.bureau_balance = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()

        # Create dataframe for aggregation
        self.agg_bureau_balance = pd.DataFrame(self.bureau_balance.SK_ID_BUREAU.unique(), columns=['SK_ID_BUREAU'])
        self.agg_bureau_balance.sort_values('SK_ID_BUREAU', inplace=True)

        # Status column
        # Since C means closed, X means status unknown, we replace C with 0 and X with mean
        dict_for_status = { 'C': 0, '0': 1, '1': 2, '2': 3, 'X': 4, '3': 5, '4': 6, '5': 7}
        self.bureau_balance['STATUS'] = self.bureau_balance['STATUS'].map(dict_for_status)
        # We weight the status by the number of months
        self.bureau_balance['MONTHS_BALANCE'] = np.abs(self.bureau_balance['MONTHS_BALANCE'])
        self.bureau_balance['WEIGHTED_STATUS'] = self.bureau_balance.STATUS / (self.bureau_balance.MONTHS_BALANCE + 1)
        self.bureau_balance = self.bureau_balance.sort_values(by=['SK_ID_BUREAU', 'MONTHS_BALANCE'], ascending=[0, 0])

        # Aggregate by SK_ID_BUREAU
        agg = self.bureau_balance.groupby(['SK_ID_BUREAU']).agg({'MONTHS_BALANCE' : ['mean','max'],
                                                                 'STATUS' : ['mean','max','first'],
                                                                 'WEIGHTED_STATUS' : ['mean','sum','first']})
        agg.columns = [col[0] + '_' + col[1].upper() for col in agg.columns]
        # Merge with agg_bureau_balance
        self.agg_bureau_balance = pd.merge(self.agg_bureau_balance, agg, on='SK_ID_BUREAU', how='left')

        # Aggregation for each year
        aggregations_for_year = {'STATUS' : ['mean','max','last','first'],
                                 'WEIGHTED_STATUS' : ['mean','max', 'first','last']}
        aggregated_bureau_years = pd.DataFrame()
        for year in range(2):
            year_group = self.bureau_balance[self.bureau_balance['MONTHS_BALANCE'] == year].groupby('SK_ID_BUREAU').agg(aggregations_for_year)
            year_group.columns = ['_'.join(ele).upper() + '_YEAR_' + str(year) for ele in year_group.columns]
            if year == 0:
                aggregated_bureau_years = year_group
            else:
                aggregated_bureau_years = aggregated_bureau_years.merge(year_group, on = 'SK_ID_BUREAU', how = 'outer')

        aggregated_bureau_rest_years = self.bureau_balance[self.bureau_balance.MONTHS_BALANCE > year].groupby(['SK_ID_BUREAU']).agg(aggregations_for_year)
        aggregated_bureau_rest_years.columns = ['_'.join(ele).upper() + '_YEAR_REST' for ele in aggregated_bureau_rest_years.columns]

        aggregated_bureau_years = aggregated_bureau_years.merge(aggregated_bureau_rest_years, on = 'SK_ID_BUREAU', how = 'outer')
        self.agg_bureau_balance = pd.merge(self.agg_bureau_balance, aggregated_bureau_years, on = 'SK_ID_BUREAU', how = 'inner')

        self.agg_bureau_balance.fillna(0, inplace = True)

        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def execute(self):
        '''Execute the feature engineering pipeline'''

        self.load_data()
        self.feature_engineering()
        
        return self.agg_bureau_balance
    
if __name__ == '__main__':
    bureau = Bureau(file_path='data/raw/dseb63_bureau.csv', 
                    bureau_balance_filepath='data/raw/dseb63_bureau_balance.csv').execute()
    bureau.to_csv('data/processed/bureau.csv', index=False)