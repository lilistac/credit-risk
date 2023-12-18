import time
import numpy as np
import pandas as pd

class InstallmentPayment:
    def __init__(self, file_path, index_col=None, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of installment payment table
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
        self.install = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print('-'*50)
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()
        # Create dataframe for aggregation
        self.agg_install = pd.DataFrame(self.install.SK_ID_CURR.unique(), columns=['SK_ID_CURR'])
        self.agg_install.dropna(inplace=True)

        # If the payment is late, risk of default is higher
        self.install['DPD'] = self.install['DAYS_ENTRY_PAYMENT'] - self.install['DAYS_INSTALMENT']
        self.install['DPD'] = self.install['DPD'].apply(lambda x: x if x > 0 else 0)

        # If the payment is early, risk of default is lower
        self.install['DBD'] = self.install['DAYS_INSTALMENT'] - self.install['DAYS_ENTRY_PAYMENT']
        self.install['DBD'] = self.install['DBD'].apply(lambda x: x if x > 0 else 0)

        # If a borrower pays only a portion of the amount due, it may indicate default.
        self.install['PCT_INSTALMENT_PAYMENT'] = self.install["AMT_PAYMENT"] / self.install["AMT_INSTALMENT"]
        # Replace inf with nan due to division by 0
        self.install['PCT_INSTALMENT_PAYMENT'] = self.install['PCT_INSTALMENT_PAYMENT'].replace(np.inf, np.nan)

        # Aggregations 
        agg = self.install.groupby('SK_ID_CURR').agg({'SK_ID_PREV': 'nunique',
                                                'NUM_INSTALMENT_VERSION': ['sum', 'mean', 'min', 'max'],
                                                'NUM_INSTALMENT_NUMBER': ['min', 'max'],
                                                'DAYS_INSTALMENT': ['mean'],
                                                'DPD': ['mean', 'max', 'count'],
                                                'DBD': ['mean', 'min', 'count'],
                                                'AMT_INSTALMENT': ['sum', 'mean', 'min'],
                                                'AMT_PAYMENT': ['sum', 'mean', 'min'],
                                                'PCT_INSTALMENT_PAYMENT': ['mean']})
        
        agg.columns = [col[0] + '_' + col[1].upper() for col in agg.columns]
        self.agg_install = pd.merge(self.agg_install, agg, how='left', on='SK_ID_CURR')

        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def other_feature_engineering(self):
        '''
        Theses features are not contributing to the highest score
        but may give some insights for further analysis or modeling
        '''

        ins = self.install.copy()

        ins['DAYS_INSTALMENT'] = ins['DAYS_INSTALMENT'].abs()
        ins['DAYS_INSTALMENT'] = ins['DAYS_INSTALMENT'] / 30  

        ins['DAYS_ENTRY_PAYMENT'] = ins['DAYS_ENTRY_PAYMENT'].abs()
        ins['DAYS_ENTRY_PAYMENT'] = ins['DAYS_ENTRY_PAYMENT'] / 30

        # 1. Calculate the total number of on-time payments in the 'ON_TIME_FLAG' column
        ins['DATE_DAY'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DATE_DAY'] = ins['DATE_DAY'].apply(lambda x: x if x > 0 else 0)
        ins['ON_TIME_STATUS'] = ins["DATE_DAY"].apply(lambda x: 1 if x == 0 else 0)
        ins['ON_TIME_SUM'] = ins.groupby(['SK_ID_CURR'])['ON_TIME_STATUS'].transform('sum')

        # 2. Count the number of payment occurrences by ID. Then 
        ins['INS_COUNT_ID'] = ins.groupby('SK_ID_CURR')['ON_TIME_STATUS'].transform('count')
        ins['ON_TIME_RATE'] = ins['ON_TIME_SUM'] / ins['INS_COUNT_ID']

        # 3. The ratio between the amount paid (AMT_PAYMENT) and the installment amount (AMT_INSTALMENT)
        ins["INSTALMENT_PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]

        # 4. The proportion of the total payment made compared to the total installment amount for each row
        ins["TOTAL_INSTALLMENT"] = ins.groupby(["SK_ID_CURR"])["AMT_INSTALMENT"].transform("sum") 
        ins["TOTAL_PAYMENT"] = ins.groupby(["SK_ID_CURR"])["AMT_PAYMENT"].transform("sum") 
        ins['PAYMENT_RATIO'] = ins['TOTAL_PAYMENT'] /ins['TOTAL_INSTALLMENT']

        # 5. Aggregation
        ins_agg = ins.groupby('SK_ID_CURR').agg({'ON_TIME_SUM': "first", # how many times the customer has made on-time payments for the first installment in all loans
                                                'INS_COUNT_ID': "first", #  how many times the customer has borrowed for the first time
                                                'TOTAL_INSTALLMENT': "first", # sum of all the installment payments the customer has made for the first loan
                                                'TOTAL_PAYMENT': "first", # sum of all the payments the customer has made for the first loan
                                                'DATE_DAY': ["sum", "mean"]}) # the total number of days between installment payments (sum) and the average number of days between installment payments (mean)
        ins_agg.fillna(0,inplace=True)
        ins_agg.columns = ["INS_" + c[0] + "_" + c[1].upper() for c in ins_agg.columns.values.tolist()]
        ins_final = ins.groupby("SK_ID_CURR").first()[['ON_TIME_RATE', 'PAYMENT_RATIO']]
        ins_final = ins_final.merge(ins_agg, how="left", on="SK_ID_CURR")
        
        return ins_final   

    def add_prefix(self):
        self.agg_install.columns = ['INSTALL_' + col for col in self.agg_install.columns]
        self.agg_install.rename(columns={'INSTALL_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)

    def execute(self):
        '''Pipeline to build the features for installment payment table'''

        self.load_data()
        self.feature_engineering()
        self.add_prefix()

        return self.agg_install
    
if __name__ == '__main__':
    install = InstallmentPayment('data/raw/dseb63_installments_payments.csv').execute()
    install.to_csv('data/processed/installment.csv', index=False)