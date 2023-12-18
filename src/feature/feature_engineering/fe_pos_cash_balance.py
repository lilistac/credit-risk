import pandas as pd
import numpy as np
import time

class PosCashBalance:
    def __init__(self, file_path, index_col=None, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of pos cash balance table
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
        self.pos = pd.read_csv(self.file_path, index_col=self.index_col)

        if self.verbose:
            print('-'*50)
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()
        # Get the last record of each previous credit of each loan
        pos_last = self.pos.sort_values(by=['MONTHS_BALANCE']).groupby(["SK_ID_CURR", "SK_ID_PREV"]).last().reset_index()
        pos_last['FLAG_DPD_DEF'] = pos_last.SK_DPD > pos_last.SK_DPD_DEF
        pos_last['FLAG_DPD_DEF'] = np.where(pos_last['FLAG_DPD_DEF'], 1, 0)

        # Create dataframe for aggregation
        self.agg_pos = pd.DataFrame(pos_last.SK_ID_CURR.unique(), columns=['SK_ID_CURR'])

        # Count number previous credit having the day past due pass exceed tolerance
        exceed = pos_last.groupby('SK_ID_CURR')['FLAG_DPD_DEF'].sum()
        self.agg_pos = pd.merge(self.agg_pos, exceed, how='left', on='SK_ID_CURR')

        # Percentage of complete contract
        complete = self.pos[self.pos.NAME_CONTRACT_STATUS == 'Completed'].groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].count()
        self.agg_pos = pd.merge(self.agg_pos, complete, how='left', on='SK_ID_CURR')

        # Aggregations of MONTHS_BALANCE, CNT_INSTALMENT, CNT_INSTALMENT_FUTURE
        mean = self.pos.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE': ['mean', 'median', 'min', 'max'],
                                                   'CNT_INSTALMENT': ['mean', 'median', 'min', 'max'],
                                                   'CNT_INSTALMENT_FUTURE': ['mean', 'median', 'min', 'max']})
        mean.columns = [col[0] + '_' + col[1].upper() for col in mean.columns]
        self.agg_pos = pd.merge(self.agg_pos, mean, how='left', on='SK_ID_CURR')

        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def other_feature_engineering(self):
        '''
        Theses features are not contributing to the highest score
        but may give some insights for further analysis or modeling
        '''
        pos = self.pos.copy()
        pos1 = pos.sort_values(by=['MONTHS_BALANCE']).groupby(["SK_ID_CURR", "SK_ID_PREV"]).last().reset_index()
        
        # 1. Calculate the total number of paid installments: 1 if the corresponding value in the "NAME_CONTRACT_STATUS" column is "Completed", and 0 otherwise.
        pos1["NEW_STATUS"] = pos1["NAME_CONTRACT_STATUS"].apply(lambda x: 1 if x == "Completed" else 0)
        pos1["NEW_COMPLETE_COUNT"] = pos1.groupby(["SK_ID_CURR"])["NEW_STATUS"].transform("sum")

        # 2. Calculate the number of overdue payments: 1 (True) if the specified conditions are met, indicating an overdue contract.
        pos1["OVERDUE_STATUS"] = (pos1["SK_DPD_DEF"] > 0) & (pos1["CNT_INSTALMENT_FUTURE"] > 0)
        pos1["OVERDUE_STATUS"] = pos1["OVERDUE_STATUS"].astype(int)
        pos1["OVERDUE_COUNT"] = pos1.groupby(["SK_ID_CURR"])["OVERDUE_STATUS"].transform("sum")

        # 3. Aggregation
        agg_pos = pos1.groupby('SK_ID_CURR').agg({
            'MONTHS_BALANCE': ['max'], # The most recent month in the loan repayment process
            'CNT_INSTALMENT': ['min', 'max', 'mean'], # The initial installment count in the loan contract and the variability of installment counts across contracts
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'], # The remaining installment count in the loan contract and the variability of remaining installments across contracts
            # Deviations from 'mean' indicate instability in the borrower's repayment behavior

            'SK_DPD': ['max', 'sum'], 
            'SK_DPD_DEF': ['max','sum'] 
            # Higher values indicate a borrower's tendency to frequently delay repayments
        })
        agg_pos.fillna(0,inplace=True)
        agg_pos.columns = ["POS_" + c[0] + "_" + c[1].upper() for c in agg_pos.columns.values.tolist()]
        pos_final = pos1.groupby('SK_ID_CURR').first()[['NEW_STATUS', 'NEW_COMPLETE_COUNT', 'OVERDUE_STATUS', 'OVERDUE_COUNT']]
        pos_final = pos_final.merge(agg_pos, how="left", on="SK_ID_CURR")
        return pos_final                                               

    def add_prefix(self):
        self.agg_pos.columns = ['POS_CASH_' + col for col in self.agg_pos.columns]
        self.agg_pos.rename(columns={'POS_CASH_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)

    def execute(self):
        '''Pipeline to build the features for modeling'''

        self.load_data()
        self.feature_engineering()
        self.add_prefix()

        return self.agg_pos
    
if __name__ == '__main__':
    pos_cash_balance = PosCashBalance(file_path='data/raw/dseb63_pos_cash_balance.csv').execute()
    pos_cash_balance.to_csv('data/processed/pos_cash_balance.csv', index=False)