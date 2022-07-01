# Data manipulation
import pandas as pd

def feature_text_mask(df,origin_feature,text,labels):
    '''
    Add a new feature to the dataset, this new feature is binary and it is created by a text serch of the specified text.

    :param df: dataframe
    :type df: pandas.DataFrame
    :param text: This is the filter text for the new feature.
    :type text: string
    :param label: This is a binary list with labels[0] for the False case and labels[1] for the True case.
    :type label: list
    :param origin_feature: feature that is being taked into account for masking process
    :type origin_feature: string
    :param new_feature: new feature name
    :type new_feature: string

    :return: dataframe with new feature
    :rtype: pandas.core.frame.DataFrame
    '''
    temp_df = df[origin_feature].str.contains(text)
    temp_df[temp_df==True]= labels[1]
    temp_df[temp_df==False]= labels[0]
    df[text] = temp_df

    return df
   
def spatial_split(df, config, n_location=25, radius=35000):
    '''
    Function to spatial split data into training and test 
    Step 1: create n random locations 
    Step 2: for each location, find data points within the circle as testing data 
    Step 3: use all points outside circles as training data 
    :param df: input pandas.DataFrame
    :param config: configuration file 
    :param n_location: number of random locations 
    :param radius: the radius of circle 
    '''
    core_locations=df.sample(n_location,random_state=12)[['latitude(m)', 'longitud(m)']].reset_index(drop=True)
    
    for index, location in core_locations.iterrows():
        selected_test=df.loc[(df['latitude(m)']-location['latitude(m)'])**2+(df['longitud(m)']-location['longitud(m)'])**2<radius**2,]
        if index == 0: 
            df_test = selected_test
        else: 
            df_test = df_test.append(selected_test, ignore_index=False)
    
    # remove duplicated points due to overlapped areas.    
    df_train=df.loc[~df.index.isin(df_test.index.unique().tolist()),]
    df_test=df_test.drop_duplicates()
    
    # reset the index 
    df_test=df_test.reset_index(drop=True)
    df_train=df_train.reset_index(drop=True)
    # print(f'{round(df_test.shape[0]/df.shape[0]*100,3)}% data as testing data')
    y_train = df_train[config['target']].copy()
    y_test = df_test[config['target']].copy()
    X_train = df_train.drop([config['target'], 'latitude(m)', 'longitud(m)'] , axis=1)
    X_test = df_test.drop([config['target'], 'latitude(m)', 'longitud(m)'] , axis=1)
    return X_train, X_test, y_train, y_test


logging_config = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'custom': {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
            'format': '%(message)s'
        }
    },

    # Any INFO level msg will be printed to the console
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'custom',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },

    'loggers': {
        '': {  # root logger
            'level': 'DEBUG',
        },
        'Client-EnsembleBuilder': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    },
}