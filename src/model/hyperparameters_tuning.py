from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import pandas as pd
from pickle import dump
import time

class HyperparameterTuner:
    '''Class to tune hyperparameters of a model using GridSearchCV and RandomizedSearchCV'''
    
    def __init__(self, param_grid, cv=None, model=LogisticRegression(), scoring='roc_auc', dataset_path='/kaggle/input/clean1712/train.csv'):
        '''
        Parameters
        ----------
        param_grid : dict
            Dictionary of hyperparameters
        cv : int, optional
            Number of folds, by default None
        model : sklearn model, optional
            Model to be tuned, by default LogisticRegression()
        scoring : str, optional
            Scoring method, by default 'roc_auc'
        dataset_path : str, optional
            File path of dataset, by default 'data/processed/train.csv'
        '''

        self.param_grid = param_grid
        self.cv = cv
        self.model = model
        self.scoring = scoring
        self.best_model = {}
        self.dataset_path = dataset_path

    def load_data(self):
        start = time.time()

        data = pd.read_csv(self.dataset_path, index_col='SK_ID_CURR')
        target = pd.read_csv('data/raw/dseb63_application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
        data = pd.merge(target, data, on='SK_ID_CURR', how='right')

        self.X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        self.y = data['TARGET']

        print('Data loaded')
        print('Time taken: ', time.time() - start)

    def split_data(self, test_size=0.2, random_state=42):
        '''
        Split data into training and validation set
        
        Parameters
        ----------
        test_size : float, optional
            Size of validation set, by default 0.2
        random_state : int, optional
            Random state, by default 42
        '''
        start = time.time()
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.X, 
                                                                              self.y, 
                                                                              test_size=test_size, 
                                                                              random_state=random_state)
        print('Data splitted')
        print('Time taken: ', time.time() - start)

    
    def resample(self):
        '''Resample training set using SMOTE'''
        start = time.time()

        self.x_train, self.y_train = SMOTE().fit_resample(self.x_train, self.y_train)

        print('Data resampled')
        print('Time taken: ', time.time() - start)

    def random_search(self):
        # If cv is not specified, use StratifiedKFold with 5 splits
        if not self.cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

        random_search = RandomizedSearchCV(self.model, 
                                           self.param_grid, 
                                           cv=skf, 
                                           scoring=self.scoring,
                                           verbose=4)
        
        random_search.fit(self.x_train, self.y_train)
        self.best_model['random_search'] = random_search.best_estimator_

    def grid_search(self):
        # If cv is not specified, use StratifiedKFold with 5 splits
        if not self.cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

        grid_search = GridSearchCV(self.model, 
                                   self.param_grid, 
                                   cv=skf, 
                                   scoring=self.scoring,
                                   verbose=4)
        
        grid_search.fit(self.x_train, self.y_train)
        self.best_model['grid_search'] = grid_search.best_estimator_

    def evaluate(self):
        '''Find the best model and evaluate it'''
        
        auc_max = 0
        model = ''
        for model in self.best_model:
            lr = self.best_model[model]
            y_pred = lr.predict_proba(self.x_val)[:][:,1]
            auc = roc_auc_score(self.y_val, y_pred)
            if auc_max < auc:
                auc_max = auc
                model = lr

        print('Best model parameters: ', model.get_params())
        print('Training score: ', auc_max*2-1)

        y_val_predict = model.predict_proba(self.x_val)[:][:,1]
        score_val = roc_auc_score(self.y_val, y_val_predict)
        print('Validation score: ', score_val*2-1)

        dump(self.best_model, open('best_model_class.pkl', 'wb'))

    def find(self):
        '''Pipeline to find the best model'''
        
        self.load_data()
        self.split_data()
        self.resample()
        self.random_search()
        self.grid_search()
        self.evaluate()
        print(self.best_model)