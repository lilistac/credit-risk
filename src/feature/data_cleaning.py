from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Cleaning:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def replace_inf(self):
        '''Function to replace infinity values with NaN due to division by 0'''
        self.train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def fill_missing_values(self):
        '''
        Function to fill missing values in numerical columns with mean and
        categorical columns with 'Missing'
        '''

        num_cols = self.train_df.select_dtypes(include='number').columns
        num_cols = num_cols.drop('SK_ID_CURR')
        cate_cols = self.train_df.select_dtypes(include='object').columns

        # # Fill missing values in numerical columns
        # imputer = SimpleImputer(strategy='mean')
        # imputer.fit(self.train_df[num_cols])
        # self.train_df[num_cols] = imputer.transform(self.train_df[num_cols])
        # self.test_df[num_cols] = imputer.transform(self.test_df[num_cols])

        # self.train_df[num_cols] = self.train_df[num_cols].astype(float)
        # self.test_df[num_cols] = self.test_df[num_cols].astype(float)

        self.train_df[num_cols] = self.train_df[num_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
        self.test_df[num_cols] = self.test_df[num_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
       
        # Fill missing values in categorical columns
        self.train_df[cate_cols] = self.train_df[cate_cols].apply(lambda x: x.fillna('Missing'), axis=0)
        self.test_df[cate_cols] = self.test_df[cate_cols].apply(lambda x: x.fillna('Missing'), axis=0)

    def remove_outliers(self, verbose=False):
        '''Function to remove outliers in numerical columns with 3 standard deviations'''

        num_cols = self.train_df.select_dtypes(include='number').columns

        for col in num_cols:
            mean, std = self.train_df[col].mean(), self.train_df[col].std()
            cut_off = std * 3
            lower, upper = mean - cut_off, mean + cut_off
            outliers = [x for x in self.train_df[col] if x < lower or x > upper]

            if verbose:
                print(f'Identified outliers - column {col}: {round(len(outliers) / len(self.train_df)*100, 2)}%')
            
            self.train_df[col][self.train_df[col] > upper] = upper
            self.train_df[col][self.train_df[col] < lower] = lower
            self.test_df[col][self.test_df[col] > upper] = upper
            self.test_df[col][self.test_df[col] < lower] = lower

    def execute(self):
        '''Pipeline to clean the data'''
        self.replace_inf()
        self.fill_missing_values()
        self.remove_outliers()

        return self.train_df, self.test_df
    
if __name__ == '__main__':
    train_df = pd.read_csv('data/preprocessed/train.csv')
    test_df = pd.read_csv('data/preprocessed/test.csv')
    train_df, test_df = Cleaning(train_df, test_df).execute()
    print(train_df.head())
    print(test_df.head())
    

