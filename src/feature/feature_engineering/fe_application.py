import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

class Application:
    '''Feature engineering for application table'''

    def __init__(self, file_path, index_col=0, verbose=True):
        '''
        Parameters
        ----------
        file_path : str
            File path of application table
        index_col : int, optional
            Index column, by default 0
        verbose : bool, optional
            Whether to print out the progress, by default True
        '''

        self.file_path = file_path
        self.index_col = index_col
        self.verbose = verbose

    def load_data(self):
        start_time = time.time()
        self.df = pd.read_csv(self.file_path, index_col=self.index_col)
        self.df.set_index('SK_ID_CURR', inplace=True)

        if self.verbose:
            print(f'Data loaded from {self.file_path}')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def data_cleaning(self):
        start_time = time.time()

        # Replace XNA values with np.nan
        self.df.replace('XNA', np.nan, inplace=True)

        # Drop columns with more than 30% missing values except 'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'EXT_SOURCE_1'
        except_df = self.df[['OWN_CAR_AGE', 'EXT_SOURCE_1']]
        self.df = self.df.dropna(thresh=len(self.df)*0.7, axis=1)
        self.df[['OWN_CAR_AGE', 'EXT_SOURCE_1']] = except_df[['OWN_CAR_AGE', 'EXT_SOURCE_1']]

        # Replace 365243 with np.nan in DAYS_EMPLOYED column
        self.df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        # Remove high imbalance columns
        cate_cols = self.df.select_dtypes(include='object').columns
        for col in cate_cols:
            if self.df[col].value_counts(normalize=True).sort_values(ascending=False)[0] > 0.95:
                self.df.drop(col, axis=1, inplace=True)

        self.df.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)

        if self.verbose:
            print(f'Cleaning data')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def feature_engineering(self):
        start_time = time.time()

        # Create new features
        # Employment duration as a percentage of life predicts loan payment ability.
        self.df['DAYS_EMPLOYED_PCT'] = self.df['DAYS_EMPLOYED'] / self.df['DAYS_BIRTH']

        # Having a large amount of available credit compared to their income may affect their ability to repay loans
        self.df['CREDIT_INCOME_PCT'] = self.df['AMT_CREDIT'] / self.df['AMT_INCOME_TOTAL']

        # An annuity provides a stable income stream (Higher annuities reduce the risk of default)
        self.df['ANNUITY_INCOME_PCT'] = self.df['AMT_ANNUITY'] / self.df['AMT_INCOME_TOTAL']

        # If it's a high percentage of available credit, the person is likely to pay off debts
        self.df['CREDIT_TERM'] = self.df['AMT_ANNUITY'] / self.df['AMT_CREDIT']

        # Polynomial features of high correlation features with target
        high_cor = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']
        df_high_cor = self.df[high_cor]

        imputer = SimpleImputer(missing_values=np.nan,strategy='median')
        df_high_cor = imputer.fit_transform(df_high_cor)

        poly = PolynomialFeatures(degree=3)
        df_high_cor = poly.fit_transform(df_high_cor)
        high_cor_cols = poly.get_feature_names_out(high_cor)
        df_high_cor = pd.DataFrame(df_high_cor, columns=high_cor_cols, index=self.df.index)

        df_high_cor.drop(['1', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'], axis=1, inplace=True) 
        self.df = pd.merge(self.df, df_high_cor, on='SK_ID_CURR', how='left')

        # New features based on External 
        self.df['EXT_SOURCES_PROD'] = self.df['EXT_SOURCE_1'] * self.df['EXT_SOURCE_2'] * self.df['EXT_SOURCE_3']
        self.df['EXT_SOURCES_WEIGHTED'] = self.df['EXT_SOURCE_1'] * 2 + self.df['EXT_SOURCE_2'] * 1 + self.df['EXT_SOURCE_3'] * 3

        # Matching credit request and purchase price is a good sign as it shows clear and reasonable intention
        self.df['CREDIT_TO_GOODS_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_GOODS_PRICE']

        # Higher income improves monthly debt payments
        self.df['INCOME_TO_EMPLOYED_RATIO'] = self.df['AMT_INCOME_TOTAL'] / self.df['DAYS_EMPLOYED']
        self.df['INCOME_TO_BIRTH_RATIO'] = self.df['AMT_INCOME_TOTAL'] / self.df['DAYS_BIRTH']

        # Time ratios
        # A higher value may indicate that the ID was changed/issued later in life, which could result in higher credit risk
        self.df['ID_TO_BIRTH_RATIO'] = self.df['DAYS_ID_PUBLISH'] / self.df['DAYS_BIRTH']

        # A higher ratio may suggest greater financial stability and affluence, as it could indicate a higher income or better financial standing
        self.df['CAR_TO_BIRTH_RATIO'] = self.df['OWN_CAR_AGE'] / self.df['DAYS_BIRTH']

        # A higher ratio may indicate that the client has been working for a longer time, which may show greater financial stability and could therefore potentially result in a lower credit risk
        self.df['CAR_TO_EMPLOYED_RATIO'] = self.df['OWN_CAR_AGE'] / self.df['DAYS_EMPLOYED']

        if self.verbose:
            print(f'Creating features')
            print(f'Elapsed time: {time.time()-start_time:.2f} seconds')
            print('-'*50)

    def add_prefix(self):   
        self.df.columns = ['APP_' + col for col in self.df.columns]
        self.df.rename(columns={'APP_SK_ID_CURR': 'SK_ID_CURR', 'APP_TARGET': 'TARGET'}, inplace=True)

    def execute(self):
        '''Execute the feature engineering pipeline'''
        
        self.load_data()
        self.data_cleaning()
        self.feature_engineering()
        self.add_prefix()

        return self.df

if __name__ == '__main__':
    app_train = Application(file_path='data/raw/dseb63_application_train.csv').execute()
    app_train.to_csv('data/processed/app_train.csv')

    app_test = Application(file_path='data/raw/dseb63_application_test.csv').execute()
    app_test.to_csv('data/processed/app_test.csv')