'''Utility functions for Explainatory Data Analysis'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix
from IPython.display import display

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def load_data(filename, index_col=None, verbose=0, filename_train=None, filename_test=None, path='../data/raw/', merge_sk_id_curr_filename=None):
    '''
    Load data and print some basic information about it

    Parameters
    ----------
    filename : str
        Name of the file to load
    index_col : str, optional
        Name of the column to use as index, by default None
    verbose : int, optional
        Level of verbosity, by default 0
        if verbose = 1: print shape, number of categorical and numerical variables, 
                        number of unique values of SK_ID_CURR, number of duplicate values
        if verbose = 2: print first 5 rows of the dataframe
        if verbose = 3: print info of the dataframe
    filename_train : str, optional
        Name of the train file, by default None
    filename_test : str, optional
        Name of the test file, by default None
    path : str, optional
        Path to the file, by default '../data/raw/'
    merge_sk_id_curr_filename : str, optional
        Name of the file to add SK_ID_CURR column, by default None
    
    Returns
    -------
    df : DataFrame
        Dataframe containing the loaded data
    
    Example
    --------    
    >>> df = load_data('application_train.csv', index_col='SK_ID_CURR', key='SK_ID_CURR', verbose=1)
    '''
    df = pd.read_csv(path+filename, index_col=index_col)

    if merge_sk_id_curr_filename:
        merge_sk_id_curr = pd.read_csv(path+merge_sk_id_curr_filename)
        df['SK_ID_CURR'] = merge_sk_id_curr['SK_ID_CURR']
        
    if verbose >= 1:
        print(f'Table: {filename}')
        print('Shape:', df.shape)
        
        print('\nNumber of categorical variables:', len(df.select_dtypes('object').columns))
        print('Number of numerical variables:', len(df.select_dtypes('number').columns))
        
        print('\nNumber of unique values of SK_ID_CURR:', df['SK_ID_CURR'].nunique()) 
        if 'SK_ID_PREV' in df.columns:
            print('Number of unique values of SK_ID_PREV:', df['SK_ID_PREV'].nunique()) 
        print('\nNumber of duplicate values:', df.duplicated().sum()) 
        
        if filename_train:
            train = pd.read_csv(path+filename_train)
            df = pd.merge(df, train[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')   
            print('\nShape of train set:', train.shape)
            overlap_id_train = set(df['SK_ID_CURR']) & set(train['SK_ID_CURR'])
            print(f'Number of overlapping SK_ID_CURR in {filename} and {filename_train}:', len(overlap_id_train))
                             
        if filename_test:
            test = pd.read_csv(path+filename_test)
            print('\nShape of test set:', test.shape)
            overlap_id_test = set(df['SK_ID_CURR']) & set(test['SK_ID_CURR'])
            print(f'Number of overlapping SK_ID_CURR in {filename} and {filename_test}:', len(overlap_id_test))
                            
    if verbose >= 2:
        display(df.head())
        
    if verbose >= 3:
        df.info(show_counts=True)
    return df 

def create_df_missing(df):
    '''
    Create a dataframe containing the number and percentage of missing values for each column

    Parameters
    ----------
    df : DataFrame
        Dataframe to check for missing values

    Returns
    -------
    df_missing : DataFrame
        Dataframe containing the number and percentage of missing values for each column

    Example
    --------
    >>> df_missing = create_df_missing(df)
    '''
    num_missing = df.isnull().sum()
    pct_missing = round(num_missing / len(df) * 100, 2)
    df_missing = pd.DataFrame({"num_missing": num_missing,
                               "pct_missing": pct_missing})
    df_missing = df_missing.sort_values(by='num_missing', ascending=False)
    return df_missing

def missing_info(df_missing, thresh=30):
    '''
    Print information about the missing values in the dataframe

    Parameters
    ----------
    df_missing : DataFrame
        Dataframe containing the number and percentage of missing values for each column
    thresh : int, optional
        Threshold for the percentage of missing values, by default 30

    Example
    --------
    >>> missing_info(df_missing, thresh=30)
    '''
    num_features = len(df_missing)
    num_more_than_thesh = len(df_missing[df_missing['pct_missing'] > thresh])
    num_not_missing = len(df_missing[df_missing['pct_missing'] == 0])
    num_few_missing = num_features - num_more_than_thesh - num_not_missing
    max_missing = round(df_missing['pct_missing'].max())
    
    print(f'There are {num_features} features:')
    print(f'- {num_not_missing} features have no missing data at all')
    print(f'- {num_few_missing} features have less than {thresh}% missing data')
    print(f'- {num_more_than_thesh} features have {thresh}-{max_missing}% missing data')
    

def plot_df_missing(df_missing, figsize=(12,4), thresh=None, x_title=0, x_subtitle=0, show_grid=True, text=True, title_size=24):
    '''
    Plot bar chart of the percentage of missing values for each column

    Parameters
    ----------
    df_missing : DataFrame
        Dataframe containing the number and percentage of missing values for each column
    figsize : tuple, optional
        Figure size, by default (12,4)
    thresh : int, optional
        Threshold for the percentage of missing values, by default None
    x_title : int, optional
        x coordinate of the title, by default 0
    x_subtitle : int, optional
        x coordinate of the subtitle, by default 0
    show_grid : bool, optional
        If True, show grid, by default True
    text : bool, optional
        If True, show subtitle, by default True
    title_size : int, optional
        Font size of the title, by default 24

    Example
    --------
    >>> plot_df_missing(df_missing, figsize=(12,4), thresh=30, x_title=0, x_subtitle=0, show_grid=True, text=True)
    '''
    df_missing = df_missing[df_missing.num_missing > 0]
    num_more_than_thesh = len(df_missing[df_missing['pct_missing'] > thresh])

    if df_missing.num_missing.sum() > 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df_missing.index, df_missing.pct_missing, color='#377349')
        
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.tick_params(left=False)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_ylim(0, 100)
        ax.set_title('Missing values', size=title_size, pad=40.5, weight='bold', x=x_title)
        if text:
            t = f'The data frame has missing values in {len(df_missing)} columns '
            if thresh:
                ax.plot(df_missing.index, [thresh] * len(df_missing), color='k')
                t += f'and {num_more_than_thesh} of them have more than {thresh}% missing data'
            ax.text(x=x_subtitle, y=110, s=t, ha='left', wrap=True, size=12)  
        
        if show_grid:
            ax.yaxis.grid(linestyle='-', alpha=0.5, zorder=1) 
            ax.xaxis.grid(linestyle='-', alpha=0.5, zorder=1)  
            ax.set_axisbelow(True)

        plt.show()
        
    else:
        print('The dataframe does not contain any missing values')

def validate_dtype(df, define=True, thresh=4):
    '''
    Validate the data type of each column and change the dtype of columns having less than 
    thresh nunique and numeric dtype to object

    Parameters
    ----------
    df : DataFrame
        Dataframe to validate
    define : bool, optional
        If True, return the dataframe containing categorical and numerical columns, 
        by default True
    thresh : int, optional
        Threshold for the number of unique values, by default 4

    Returns
    -------
    df : DataFrame
        Dataframe with changed dtype of columns having less than thresh nunique and 
        numeric dtype to object
    cate : DataFrame
        Dataframe containing categorical columns of the original dataframe
    num : DataFrame
        Dataframe containing numerical columns of the original dataframe

    Example
    --------
    >>> df, cate, num = validate_dtype(df, define=True, thresh=4)
    '''
    num = df.select_dtypes(include='number')

    # Find columns having less than thresh nunique and numeric dtype
    cate_cols = [col for col in num.columns if num[col].nunique() < thresh]

    # Change above columns' dtype to object
    df[cate_cols] = df[cate_cols].astype(object)
    
    if define:
        cate = df.select_dtypes(include='object')
        num = df.select_dtypes(include='number')
        return df, cate, num
    
    return df

def describe_class(cate_df):
    '''Describe the frequency of each category in the dataframe'''

    df_describe = cate_df.describe().T
    df_describe['pct_freq'] = df_describe['freq'] / df_describe['count'] * 100
    df_describe.sort_values('pct_freq', ascending=False, inplace=True)
    return df_describe

def plot_imbalance(df_describe, figsize=(12,4), thresh=0, x_title=0, show_grid=True, title_size=24):
    '''
    Plot bar chart of the percentage of top frequency for each column

    Parameters
    ----------
    df_describe : DataFrame
        Dataframe containing the frequency of each category in the dataframe
    figsize : tuple, optional
        Figure size, by default (12,4)
    thresh : int, optional
        Threshold for the percentage of top frequency, by default 0
    x_title : int, optional
        x coordinate of the title, by default 0
    show_grid : bool, optional
        If True, show grid, by default True
    title_size:
        Font size of the title, by default 24
    '''

    df_describe = df_describe[df_describe.pct_freq > 0]
    pct_more_than_thesh = len(df_describe[df_describe['pct_freq'] > thresh])

    if pct_more_than_thesh > 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df_describe.index, df_describe.pct_freq, color='#377349')
        
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.tick_params(left=False)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_ylim(0, 100)
        ax.set_title('Top frequency percentage', size=title_size, pad=40, weight='bold', x=x_title)
        
        if show_grid:
            ax.yaxis.grid(linestyle='-', alpha=0.5, zorder=1)  
            ax.xaxis.grid(linestyle='-', alpha=0.5, zorder=1)  
            ax.set_axisbelow(True)

        plt.show()
    else:
        print('The dataframe does not contain any inbalanced data')

def plot_boxplot(df, figsize=(20, 40), color='#377349'):
    '''
    Plot boxplot for each column of the dataframe

    Parameters
    ----------
    df : DataFrame
        Dataframe to plot
    figsize : tuple, optional
        Figure size, by default (20, 40)
    color : str, optional
        Color of the boxplot, by default '#377349'

    Example
    --------
    >>> plot_boxplot(df, figsize=(20, 40), color='#377349')
    '''

    plt.figure(figsize=figsize)
    for i in range(0, df.shape[1]):
        plt.subplot(df.shape[1]//4+1,4,i+1)
        sns.boxplot(x=df.iloc[:, i], color=color)
        plt.title(df.columns[i], fontsize=20)
        plt.xlabel(' ')
        plt.tight_layout()

def day_to_year(df):
    '''
    Convert columns containing 'DAYS' to years

    Parameters
    ----------
    df : DataFrame
        Dataframe to convert

    Returns
    -------
    df : DataFrame
        Dataframe with converted columns

    Example
    --------
    >>> df = day_to_year(df)
    '''
    day_cols = df.filter(like='DAYS', axis=1).columns
    df[day_cols] = round(df[day_cols] / 365)
    df.columns = df.columns.str.replace('DAYS', 'YEARS')
    return df 

class CorrelationMatrix:
    '''
    Class to plot correlation matrix for numerical and categorical features
    and return top correlated features with the target variable

    Parameters
    ----------
    data : DataFrame
        Dataframe to plot correlation matrix
    drop_cols : list, optional
        List of columns to drop, by default None
    figsize : tuple, optional
        Figure size, by default (12,12)
    tight_layout : bool, optional
        If True, adjust the padding between and around subplots, by default True
    mask_upper : bool, optional
        If True, mask the upper triangle of the heatmap, by default True
    linewidth : float, optional
        Width of the lines that will divide each cell, by default 0.1
    fontsize : int, optional
        Font size for the x and y labels, by default 10
    cmap : str, optional
        The mapping from data values to color space, by default 'viridis'
    
    Example
    --------
    >>> cm = CorrelationMatrix(df, drop_cols=['SK_ID_CURR', 'SK_ID_PREV'])
    >>> cm.plot_correlation_matrix()
    >>> cm.plot_phik_matrix()
    >>> cm.target_top()
    >>> cm.target_top_cate()
    >>> cm.target_top_num()
    '''
    def __init__(self, data, drop_cols=None, figsize=(12,12), 
                 tight_layout=True, mask_upper=True,
                 linewidth=0.1, fontsize=10, cmap='viridis'):

        if drop_cols:
            data = data.drop(drop_cols, axis=1)
        self.data = data.dropna(subset=['TARGET'])
        self.figsize = figsize
        self.tight_layout = tight_layout
        self.mask_upper = mask_upper
        self.linewidth = linewidth
        self.fontsize = fontsize
        self.cmap = cmap
        
        self.phik_matrix = self._get_phik_matrix()
        self.corr_matrix = self._get_corr_matrix()
        
    def _get_phik_matrix(self):
        ''' Calculate the phik matrix for categorical features '''

        cate_data = self.data.select_dtypes(include='object')
        return cate_data.phik_matrix()
    
    def _get_corr_matrix(self):
        ''' Calculate the correlation matrix for numerical features '''

        num_data = self.data.select_dtypes(include='number')
        return num_data.corr()
        
    def plot_correlation_matrix(self):
        ''' Plot the correlation matrix for numerical features '''

        if self.mask_upper:
            # Masking the heatmap to show only lower triangle
            mask_array = np.ones(self.corr_matrix.shape)
            mask_array = np.triu(mask_array)
        else:
            mask_array = np.zeros(self.corr_matrix.shape)
            
        self._plot_heatmap(self.corr_matrix, mask_array, 'Correlation Heatmap for Numerical features')
        
    def plot_phik_matrix(self):
        ''' Plot the phik matrix for categorical features '''
        if self.mask_upper:
            mask_array = np.ones(self.phik_matrix.shape)
            mask_array = np.triu(mask_array)
        else:
            mask_array = np.zeros(self.phik_matrix.shape)

        self._plot_heatmap(self.phik_matrix, mask_array, 'Phi-K Correlation Heatmap for Categorical Features')
            
    def _plot_heatmap(self, matrix, mask_array, title):
        '''
        Plot the heatmap for the given matrix
        
        Parameters
        ----------
        matrix : DataFrame
            Matrix to plot
        mask_array : array
            Array to mask the upper triangle of the heatmap
        title : str
            Title of the plot
        '''
        plt.figure(figsize = self.figsize, tight_layout = self.tight_layout)
        sns.heatmap(matrix, annot=False, mask=mask_array, linewidth=self.linewidth, 
                    cmap=self.cmap, vmin=0, vmax=1)
        plt.xticks(rotation = 90, fontsize = self.fontsize)
        plt.yticks(fontsize = self.fontsize)
        plt.title(title, size=20)
        plt.show()
        
    def target_top(self, top_columns=10):     
        ''' Return top correlated features with the target variable according to phik matrix'''

        phik_target_arr = self.phik_matrix[['TARGET']].T 
        num_cols = self.data.select_dtypes(include='number').columns
        for column in num_cols:
            phik_target_arr[column] = self.data[['TARGET', column]].phik_matrix(column).iloc[0,1]
        phik_target_arr = phik_target_arr.T 
        phik_target_arr.rename(columns={'TARGET': 'PhiK-Correlation'}, inplace=True)
        return phik_target_arr.sort_values(by='PhiK-Correlation', ascending=False).drop('TARGET')[:top_columns]
    
    def target_top_cate(self, top_columns=10):  
        ''' Return top correlated categorical features with the target variable according to phik matrix''' 
        top = self.phik_matrix[['TARGET']].rename(columns={'TARGET': 'PhiK-Correlation'})
        return top.sort_values(by='PhiK-Correlation', ascending=False).drop('TARGET')[:top_columns]
    
    def target_top_num(self, top_columns=10):   
        ''' Return top correlated numerical features with the target variable according to correlation matrix'''     
        target = self.data.TARGET.astype(int)
        num_data = self.data.select_dtypes(include='number')
        num_data['TARGET'] = target 
        top = num_data.corr()[['TARGET']].abs().rename(columns={'TARGET': 'Correlation'})
        return top.sort_values(by='Correlation', ascending=False).drop('TARGET')[:top_columns]
    
def plot_categorical_variables_bar(data, column_name, figsize = (16,8), plot_defaulter = True, rotation = 90, horizontal=False):
    '''
    Plot bar chart for categorical variables

    Parameters
    ----------
    data : DataFrame
        Dataframe to plot
    column_name : str
        Name of the column to plot
    figsize : tuple, optional
        Figure size, by default (18,6)
    plot_defaulter : bool, optional
        If True, plot the percentage of defaulters for each category, by default True
    rotation : int, optional
        Rotation of the xticks, by default 90
    horizontal : bool, optional
        If True, plot horizontal bar chart, by default False
    '''
    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")
    
    plt.figure(figsize = figsize, tight_layout = False)
    sns.set(style = 'whitegrid', font_scale = 1.2)
    
    # Plotting overall distribution of category
    plt.subplot(1,2,1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending = False)
    
    x = data_to_plot.index
    y = data_to_plot
    if horizontal:
        x, y = y, x
    ax = sns.barplot(x=x, y=y, palette = 'Set1')
    
    plt.xlabel(column_name, labelpad = 10)
    plt.title(f'Distribution of {column_name}', pad = 20)
    plt.xticks(rotation = rotation)
    plt.ylabel('Counts')
    
    # Plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)

        plt.subplot(1,2,2)
        x = percentage_defaulter_per_category.index
        y = percentage_defaulter_per_category
        if horizontal:
            x, y = y, x
        sns.barplot(x=x, y=y, palette = 'Set2')
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name, labelpad = 10)
        plt.xticks(rotation = rotation)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad = 20)
    plt.show()

def plot_continuous_variables(data, column_name, plots = ['distplot', 'box'], scale_limits = None, figsize = (16,8), log_scale = False):
    '''
    Plot PDF and box plot for continuous variables

    Parameters
    ----------
    data : DataFrame
        Dataframe to plot
    column_name : str
        Name of the column to plot
    plots : list, optional
        List of plots to plot, by default ['distplot', 'box']
    scale_limits : tuple, optional
        Limits of the scale, by default None
    figsize : tuple, optional
        Figure size, by default (20,8)
    log_scale : bool, optional
        If True, plot the log scale, by default False
    '''

    data_to_plot = data.copy()
    data_to_plot.dropna(subset='TARGET')
    if scale_limits:
        # Taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    
    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'distplot':  
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 0].dropna(),
                         label='Non-Defaulters', hist = False, color='red')
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 1].dropna(),
                         label='Defaulters', hist = False, color='black')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'box':  
            sns.boxplot(x='TARGET', y=column_name, data=data_to_plot, color='#377349')
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    plt.show()     

def plot_overdue_status(dataframe):
    """
    Takes a DataFrame, calculates the overdue status for each transaction,
    counts the number of overdue transactions per customer, and plots the 
    result using the 'Set2' color palette.
    """

    # Function to determine overdue status
    def overdue_flag(x):
        if x['SK_DPD_DEF'] > 0 and x['CNT_INSTALMENT_FUTURE'] > 0:
            return 1
        else:
            return 0

    # Apply the overdue_flag function to the DataFrame
    dataframe["Overdue_flag"] = dataframe.apply(overdue_flag, axis=1)

    # Count the number of overdue transactions by SK_ID_CURR
    dataframe["POS_BL_OVERDUE_COUNT"] = dataframe.groupby(["SK_ID_CURR"])["Overdue_flag"].transform("sum")

    # Drop the helper column
    dataframe.drop(["Overdue_flag"], axis=1, inplace=True)

    # Group by the count of overdue transactions and get the size, excluding zero count
    overdue_count_SK = dataframe.groupby(["POS_BL_OVERDUE_COUNT"]).size()[1:]

    # Plotting with 'Set2' color palette
    plt.figure(figsize=(16, 5))
    sns.barplot(x=overdue_count_SK.index, y=overdue_count_SK.values, palette="Set2")
    plt.xlabel("Number of Overdue Transactions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Overdue Transactions per Customer")
    plt.show()

def plot_completed_status(dataframe):
    """
    Takes a DataFrame, processes it to count the number of completed transactions,
    and plots the result using a pastel color palette.
    """
    # Filter and create a flag for completed status
    dataframe["Completed_Flag"] = dataframe["NAME_CONTRACT_STATUS"].apply(lambda x: 1 if x == "Completed" else 0)
    
    # Count the number of completed transactions by SK_ID_CURR
    dataframe["POS_BL_COMPLETE_COUNT"] = dataframe.groupby(["SK_ID_CURR"])["Completed_Flag"].transform("sum")
    
    # Drop the helper column
    dataframe.drop(["Completed_Flag"], axis=1, inplace=True)

    # Group by the count of completed transactions and get the size
    completed_count_SK = dataframe.groupby(["POS_BL_COMPLETE_COUNT"]).size()

    # Plotting with a pastel color palette
    plt.figure(figsize=(16, 5))
    sns.barplot(x=completed_count_SK.index, y=completed_count_SK.values, palette="pastel")
    plt.xlabel("Number of Completed Transactions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Completed Transactions per Customer")
    plt.show()