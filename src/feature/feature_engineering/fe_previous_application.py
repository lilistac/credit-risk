import pandas as pd
import numpy as np
import time

def find_mode(x):
    '''Function to find mode of a series'''
    if len(x.mode()) > 1:
        return x.mode()[0]
    elif len(x.mode()) == 1:
        return x.mode()
    else:
        return np.nan

class PreviousApplication:
    def __init__(self, file_path, index_col=None, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of previous application table
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
        self.prev = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def data_cleaning(self):
        start_time = time.time()
        # Replace XNA values with np.nan
        self.prev.replace('XNA', np.nan, inplace=True)

        # Replace 365243 with np.nan in days column
        days = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_TERMINATION']
        for col in days:
            self.prev[col].replace(365243, np.nan, inplace=True)

        if self.verbose:
            print(f'Data cleaning')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()
        # Create dataframe for aggregation
        self.agg_prev = pd.DataFrame(self.prev.SK_ID_CURR.unique(), columns=['SK_ID_CURR'])

        # Aggregations for numerical columns
        num = self.prev.groupby('SK_ID_CURR').agg({'AMT_ANNUITY': ['min', 'max', 'sum', 'mean'],
                                                   'AMT_CREDIT': ['min', 'max', 'sum', 'mean'],
                                                   'AMT_APPLICATION': ['min', 'max', 'sum', 'mean'],
                                                   'AMT_DOWN_PAYMENT': ['min', 'max', 'sum', 'mean'],
                                                   'AMT_GOODS_PRICE': ['min', 'max', 'sum', 'mean'],
                                                   'CNT_PAYMENT': ['min', 'max', 'sum', 'mean'],
                                                    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                                                    'DAYS_DECISION': ['min', 'max', 'mean'], 
                                                    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean']})
        num.columns = [col[0] + '_' + col[1].upper() for col in num.columns]
        self.agg_prev = pd.merge(self.agg_prev, num, how='left', on='SK_ID_CURR')

        # Mode for categorical columns
        cate = list(self.prev.select_dtypes(include=object).columns)
        cate_col = self.prev.groupby('SK_ID_CURR')[cate].agg(lambda x: find_mode(x))
        cate_col.columns = [col.upper() + '_MODE' for col in cate_col.columns]
        self.agg_prev = pd.merge(self.agg_prev, cate_col, how='left', on='SK_ID_CURR')

        # Keep the last record for days columns
        days = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_TERMINATION']
        for day in days:
            last = self.prev.sort_values(day).groupby('SK_ID_CURR')[[day]].last()
            last.columns = [day + '_LAST']
            self.agg_prev = pd.merge(self.agg_prev, last, how='left', on='SK_ID_CURR')

        # Number of flags
        flag = self.prev.groupby('SK_ID_CURR')[['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_INSURED_ON_APPROVAL']].count()
        flag.columns = ['CNT' + col for col in flag.columns]
        self.agg_prev = pd.merge(self.agg_prev, flag, how='left', on='SK_ID_CURR')

        # Create new features and aggregations
        #The percentage of credit requested compared to the amount of credit actually granted
        self.prev['PERCENTAGE_CREDIT'] = self.prev['AMT_APPLICATION'] / self.prev['AMT_CREDIT']

        # The ratio of the credit amount to the annuity amount
        self.prev['PAYMENT_RATE'] = self.prev['AMT_CREDIT'] / self.prev['AMT_ANNUITY']

        # The percentage of the application amount compared to the goods price for each loan application
        self.prev['PERCENTAGE_APPLICATION_GOODS_RATE'] = self.prev['AMT_APPLICATION'] / self.prev['AMT_GOODS_PRICE']

        # The percentage of the credit amount compared to the goods price for each loan application
        self.prev['PERCENTAGE_CREDIT_GOODS'] = self.prev['AMT_CREDIT'] / self.prev['AMT_GOODS_PRICE']
        
        # The difference (measured in days) between the decision date and the estimated return day of a loan
        # A higher value of RETURN_DAY indicates a longer duration for the loan
        self.prev['RETURN_DAY'] = self.prev['DAYS_DECISION'] + self.prev['CNT_PAYMENT'] * 30

        # The difference between the termination date and the estimated return day
        self.prev['DAYS_TERMINATION_DIFF'] = self.prev['DAYS_TERMINATION'] - self.prev['RETURN_DAY']

        # The differences between the last due date (1st version) and the first due date
        self.prev['DAYS_DUE_DIFF'] = self.prev['DAYS_LAST_DUE_1ST_VERSION'] - self.prev['DAYS_FIRST_DUE']

        # If the value of x is less than or equal to 1, the customer has obtained the desired loan amount or a larger loan
        # Indicates whether the customer has obtained the desired loan amount or a larger loan.
        self.prev["APP_CREDIT_RATE_RATIO"] = self.prev["PERCENTAGE_CREDIT"].apply(lambda x: 1 if(x<=1) else 0)

        # The classification of the number of payment installments of loans into three categories: short (0-12), medium (12-60), and long (60-120)
        self.prev["NEW_CNT_PAYMENT"]=pd.cut(x=self.prev['CNT_PAYMENT'], bins=[0, 12, 60,120], labels=["Short", "Medium", "Long"])

        # The difference between the estimated end date and the final due date
        self.prev["NEW_END_DIFF"] = self.prev["DAYS_TERMINATION"] - self.prev["DAYS_LAST_DUE"]

        # If the value in the "PERCENTAGE_CREDIT" column is less than or equal to 1
        # it means the customer has obtained the desired loan amount or borrowed more than the desired amount
        # and the corresponding value in the "APP_CREDIT_RATE_RATIO" column will be 1
        self.prev["APP_CREDIT_RATE_RATIO_0"] = self.prev["APP_CREDIT_RATE_RATIO"].value_counts()[0]

        # If the value in the "PERCENTAGE_CREDIT" column is greater than 1
        # it means the customer has borrowed less than the desired amount
        # and the corresponding value in the "APP_CREDIT_RATE_RATIO" column will be 0
        self.prev["APP_CREDIT_RATE_RATIO_1"] = self.prev["APP_CREDIT_RATE_RATIO"].value_counts()[1]

        agg = self.prev.groupby('SK_ID_CURR').agg({'PERCENTAGE_CREDIT': ['max', 'min', 'mean'],
                                      'PAYMENT_RATE': ['max', 'min', 'mean'],
                                      'PERCENTAGE_APPLICATION_GOODS_RATE': ['max', 'min', 'mean'],
                                      'PERCENTAGE_CREDIT_GOODS': ['max', 'min', 'mean'],
                                      'RETURN_DAY': ['max', 'min', 'mean'],
                                      'DAYS_TERMINATION_DIFF': ['max', 'min', 'mean'],
                                      'DAYS_DUE_DIFF': ['max', 'min', 'mean'],
                                      'APP_CREDIT_RATE_RATIO': ['mean'],
                                      'APP_CREDIT_RATE_RATIO_1': ['sum'],
                                      'APP_CREDIT_RATE_RATIO_0': ['sum'],
                                      'NEW_CNT_PAYMENT': [find_mode],
                                      'NEW_END_DIFF': ['max', 'min', 'mean']})
        
        agg.columns = [col[0] + '_' + col[1].upper() for col in agg.columns]

        self.agg_prev = pd.merge(self.agg_prev, agg, how='left', on='SK_ID_CURR')

        if self.verbose:    
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def add_prefix(self):
        self.agg_prev.columns = ['PREV_' + col for col in self.agg_prev.columns]
        self.agg_prev.rename(columns={'PREV_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)

    def execute(self):
        '''Pipeline to build the features for modeling'''

        self.load_data()
        self.data_cleaning()
        self.feature_engineering()
        self.add_prefix()

        return self.agg_prev
    
if __name__ == '__main__':
    prev = PreviousApplication('data/raw/dseb63_previous_application.csv').execute()
    prev.to_csv('data/processed/previous_application.csv', index=False)