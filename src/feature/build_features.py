import json
from utils import merge, encode
import pandas as pd 
from data_cleaning import Cleaning
from feature_engineering import Binning, Application, Bureau, CreditCardBalance, InstallmentPayment, PosCashBalance, PreviousApplication

class FeatureBuilder:
    '''Preparation of data for modeling'''

    def __init__(self, file_direrctory='data/raw/', bins_direrctory='src/feature/feature_engineering/'):
        '''
        Parameters
        ----------
        file_direrctory : str, optional
            Directory of the raw data, by default 'data/raw/'
        bins_direrctory : str, optional
            Directory of the bins, by default 'src/feature/feature_engineering
        '''

        self.file_direrctory = file_direrctory
        self.bins_direrctory = bins_direrctory
        self.df_train = None
        self.df_test = None

    def merge_data(self):
        '''
        Function to merge all the tables together with the application_train and application_test tables
        on SK_ID_CURR.
        '''
        app_train = Application(file_path=self.file_direrctory+'dseb63_application_train.csv').execute()
        app_test = Application(file_path=self.file_direrctory+'dseb63_application_test.csv').execute()
        bureau = Bureau(file_path=self.file_direrctory+'dseb63_bureau.csv', bureau_balance_filepath=self.file_direrctory+'dseb63_bureau_balance.csv').execute()
        credit_card_balance = CreditCardBalance(file_path=self.file_direrctory+'dseb63_credit_card_balance.csv').execute()
        installments_payments = InstallmentPayment(file_path=self.file_direrctory+'dseb63_installments_payments.csv').execute()
        pos_cash_balance = PosCashBalance(file_path=self.file_direrctory+'dseb63_pos_cash_balance.csv').execute()
        previous_application = PreviousApplication(file_path=self.file_direrctory+'dseb63_previous_application.csv').execute()

        self.df_train = merge(app_train, bureau, credit_card_balance, installments_payments, pos_cash_balance, previous_application)
        self.df_test = merge(app_test, bureau, credit_card_balance, installments_payments, pos_cash_balance, previous_application)
        
        self.df_train.set_index('SK_ID_CURR', inplace=True)
        self.df_test.set_index('SK_ID_CURR', inplace=True)
        
        self.target = self.df_train[['TARGET', 'SK_ID_CURR']]
        self.df_train.drop(['TARGET'], axis=1, inplace=True)

    def clean(self):
        clean = Cleaning(self.df_train, self.df_test)
        self.df_train, self.df_test = clean.execute()
        self.df_train['TARGET'] = pd.merge(self.target, self.df_train, on='SK_ID_CURR', how='right')

        # Save cleaned data
        self.df_train.to_csv('data/processed/train_clean.csv')
        self.df_test.to_csv('data/processed/test_clean.csv')

    def binning(self):
        '''
        Function using binning to convert numerical features into categorical features,
        encode features and select features with IV >= 0.02.
        '''

        with open(self.bins_direrctory+'bins.json') as f:
            bins = json.load(f)

        data_binning = Binning(self.df_train, self.target)
        df_IV = data_binning.create_IV_info(bins=bins)

        select_feature = df_IV[df_IV['IV']>=0.02]['Features'].values
        self.df_train = self.df_train[select_feature]
        self.df_test = self.df_test[select_feature]

        self.df_train = encode(self.df_train, bins)
        self.df_test = encode(self.df_test, bins)
        
    def build(self, to_csv=False, to_directory='data/processed/'):
        '''Pipeline to build the features for modeling'''

        self.merge_data()
        self.clean()
        self.binning()

        print(f'Train shape: {self.df_train.shape}')
        print(f'Test shape: {self.df_test.shape}')

        if to_csv:
            self.df_train.to_csv(to_directory+'train.csv')
            self.df_test.to_csv(to_directory+'test.csv')

        return self.df_train, self.df_test
    
if __name__ == '__main__':
    feature_builder = FeatureBuilder()
    feature_builder.build(to_csv=True)
