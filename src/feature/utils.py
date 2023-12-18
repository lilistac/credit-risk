import pandas as pd
import numpy as np

def merge(application, bureau, credit, install, pos, prev):
    '''
    Function to merge all the tables together with the application_train and application_test tables
    on SK_ID_CURR.
    
    Parameters
    ----------
    All the previously pre-processed Tables.
        
    Returns
    -------
    Single merged tables, one for training data and one for test data
    '''

    df = pd.merge(application, bureau, on='SK_ID_CURR', how='left')
    df = pd.merge(df, credit, on='SK_ID_CURR', how='left')
    df = pd.merge(df, install, on='SK_ID_CURR', how='left')
    df = pd.merge(df, pos, on='SK_ID_CURR', how='left')
    df = pd.merge(df, prev, on='SK_ID_CURR', how='left')

    return df

def encode(df, bins):
    '''
    Function to encode categorical features into numerical features
    
    Parameters
    ----------
    df : dataframe
        Dataframe to be encoded
    bins : dict
        Dictionary of bins for each feature
    '''
    for col in df:
            num_bins = len(bins[col])
            for i in range(num_bins-1):
                name = col + '_' + str(i)
                if df[col].dtype == 'object':
                    values = df[col].unique()
                    for val in values:
                        if val not in bins[col]:
                            df.replace(val, 'Other', inplace=True)
                    df[name] = np.where(df[col]==bins[col][i], 1, 0)
                else:
                    df[name] = np.where((df[col]>=bins[col][i]) & (df[col]<bins[col][i+1]), 1, 0)
            df.drop(col, axis=1, inplace=True)
    return df    

def reduce_memory_usage(df):
    '''Function to reduce memory usage of dataframe'''
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))  
    return df