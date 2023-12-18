import pandas as pd
import pickle

class Prediction:
    def __init__(self, model, file_path='data/processed/test.csv'):
        '''
        Parameters
        ----------
        model : sklearn model
            Model to be used for prediction
        file_path : str, optional
            File path of the train set, by default 'data/processed/test.csv'
        '''
        self.model = model
        self.file_path = file_path
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.id = self.data.SK_ID_CURR
        self.data.drop('SK_ID_CURR', axis=1, inplace=True)
        
    def predict(self):
        self.y_pred = self.model.predict_proba(self.data)[:,1]
        self.y_pred = pd.DataFrame({'SK_ID_CURR': self.id, 'TARGET': self.y_pred})
        self.y_pred.to_csv('data/submission/prediction.csv', index=False)

if __name__ == "__main__":
    model = pickle.load(open('src/model/best_model.pkl', 'rb'))
    prediction = Prediction(model)
    prediction.load_data()
    prediction.predict()
