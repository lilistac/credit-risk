import pandas as pd
import time

class CreditCardBalance:
    def __init__(self, file_path, index_col=None, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of credit card balance table
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
        self.credit = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()
        # Large amount of drawings may indicate a higher defaulting tendency
        self.credit['AMT_TOTAL_DRAWINGS'] = self.credit.AMT_DRAWINGS_CURRENT + self.credit.AMT_DRAWINGS_ATM_CURRENT + self.credit.AMT_DRAWINGS_OTHER_CURRENT + self.credit.AMT_DRAWINGS_POS_CURRENT
        
        # Large number of drawings may indicate a higher defaulting tendency
        self.credit['CNT_TOTAL_DRAWINGS'] = self.credit.CNT_DRAWINGS_ATM_CURRENT + self.credit.CNT_DRAWINGS_CURRENT + self.credit.CNT_DRAWINGS_OTHER_CURRENT + self.credit.CNT_DRAWINGS_POS_CURRENT

        # Create dataframe for aggregation
        self.agg_credit = pd.DataFrame(self.credit.SK_ID_CURR.unique(), columns=['SK_ID_CURR'])

        # Keep last record
        last = self.credit.sort_values('MONTHS_BALANCE').groupby('SK_ID_CURR')[list(self.credit.columns)].last()
        last.columns = [col + '_LAST' for col in last.columns]
        self.agg_credit = pd.merge(self.agg_credit, last, how='left', on='SK_ID_CURR')

        # Aggregations for numerical columns
        num = self.credit.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE': ['min', 'max'],
                                                     'AMT_BALANCE': ['sum', 'min', 'max', 'mean'],
                                                     'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
                                                     'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'max'],
                                                     'AMT_DRAWINGS_CURRENT': ['sum', 'max'],
                                                     'AMT_DRAWINGS_OTHER_CURRENT': ['sum', 'max'],
                                                     'AMT_DRAWINGS_POS_CURRENT': ['sum', 'max'],
                                                     'AMT_INST_MIN_REGULARITY': ['mean', 'min', 'max'],
                                                     'AMT_PAYMENT_CURRENT': ['sum', 'min'],
                                                     'AMT_PAYMENT_TOTAL_CURRENT': ['sum', 'min', 'mean'],
                                                     'AMT_RECEIVABLE_PRINCIPAL': ['sum', 'max'],
                                                     'AMT_RECIVABLE': ['sum', 'max'],
                                                     'AMT_TOTAL_RECEIVABLE': ['sum', 'max', 'mean'],
                                                     'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'max'],
                                                     'CNT_DRAWINGS_CURRENT': ['mean', 'max'],
                                                     'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'max'],
                                                     'CNT_DRAWINGS_POS_CURRENT': ['mean', 'max'],
                                                     'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean'],
                                                     'SK_DPD': ['max', 'min', 'mean'],
                                                     'SK_DPD_DEF': ['max', 'min', 'mean'],
                                                     'AMT_TOTAL_DRAWINGS': ['sum', 'mean', 'min', 'max'],
                                                     'CNT_TOTAL_DRAWINGS': ['sum', 'mean', 'min', 'max']})
        num.columns = [col[0] + '_' + col[1].upper() for col in num.columns]
        self.agg_credit = pd.merge(self.agg_credit, num, how='left', on='SK_ID_CURR')

        # Mode of NAME_CONTRACT_STATUS
        # Contracts belonging to 'Demand' and 'Refused' have the highest defaulting tendency as compared to the rest
        cate = self.credit.groupby('SK_ID_CURR')[['NAME_CONTRACT_STATUS']].agg(lambda x: x.mode()[0])
        cate.columns = [col + '_MODE' for col in cate.columns]
        self.agg_credit = pd.merge(self.agg_credit, cate, how='left', on='SK_ID_CURR')

        # Completed and refused flags
        # Applicants with a history of not completing or being refused contracts may have a higher defaulting tendency
        contract_status = self.credit.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack()[['Completed', 'Refused']]
        contract_status.fillna(0, inplace=True)
        contract_status[contract_status.Completed > 0] = 1
        contract_status[contract_status.Refused > 0] = 1

        self.agg_credit['FLAG_CONTRACT_COMPLETED'] = contract_status['Completed']
        self.agg_credit['FLAG_CONTRACT_REFUSED'] = contract_status['Refused']

        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def add_prefix(self):
        self.agg_credit.columns = ['CREDIT_CARD_' + col for col in self.agg_credit.columns]
        self.agg_credit.rename(columns={'CREDIT_CARD_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)

    def execute(self):
        '''Pipeline to clean the data'''

        self.load_data()
        self.feature_engineering()
        self.add_prefix()

        return self.agg_credit
    
if __name__ == '__main__':
    credit_card = CreditCardBalance('data/raw/dseb63_credit_card_balance.csv').execute()
    credit_card.to_csv('data/processed/credit_card.csv', index=False)