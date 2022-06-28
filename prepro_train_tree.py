'''
This script defines a preprocessing pipeline that reads the dataset from a blobstore. The pipeline encodes, 
normalizes and identifies the task (regression/classification). If the identified task is classification, 
the pipeline removes the not desired categories (if specified)  and if balances the dataset (if specified).

This pipeline must be executed with a configuration file.
For instance: python prepro_train.py h1.json

'''
from  warnings import filterwarnings
filterwarnings('ignore')
import json
import joblib
import pandas as pd
import numpy as np

from sys import argv
from shutil import rmtree
from tempfile import TemporaryDirectory
from functools import partial


from azureml.core import Workspace, Experiment, Dataset
from azureml.interpret import ExplanationClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from interpret.ext.blackbox import TabularExplainer

from functions.data_processing.validation import validate_config
from functions.data_processing.preprocessing import spatial_split
from functions.models_processing.models_pipeline import lgbm_objective, lgbm_training, rf_training, rf_objective, setup_azure_workspace

def main():
    #Loads configuration file information as a dict.
    print('>> Loading configuration file...')
    if len(argv)==2:
        f = open('configs/prepro_train/' + argv[1])
    else:
        print('>> Configuration file is not specified, default.json will be used')
        f = open('configs/prepro_train/default.json')
    
    
    try:
        config = json.load(f)
    except:
        print('>> Error: The configuration file is corrupted or bad specified')
        return 0
    print('>> Done.')


    # Set up azure workspace "DevAIMLWorkspace" using service connection authentication 
    ws = setup_azure_workspace()
    
    # Change this validation with cerberus and create a function
    # Checks the ds and config file definitions (they must be well specified).

    input_ds = Dataset.get_by_name(ws, name=config['ds_name'], version=config['ds_version'])
    df = input_ds.to_pandas_dataframe()
    print(len(df.columns))
    categorical_features = list(df.select_dtypes(include=['object']).columns)
    df[categorical_features] = df[categorical_features].astype('category')
    validate_config(config,'prepro_train',df)
    

    if 'feat_filter' in config:
        if config['feat_filter']['method'] == 'select':
            df = df[config['feat_filter']['features']] 
        elif config['feat_filter']['method'] == 'remove':
            df = df.drop(config['feat_filter']['features'], axis=1)
    print(len(df.columns))
    if 'cate_remove' in config:

        if config['cate_filter']['method'] == 'delete':
            df = df[df[config['target'] not in config['cate_filter']['categories']]]
            
        elif config['cate_filter']['method'] == 'replace':
            df[config['target']] = df[config['target']].replace(config['cate_filter']['categories'],config['cate_filter']['rep_name'])
    
    obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        df[feature] = pd.Series(df[feature], dtype="category")
    
    print(len(df.columns))
    y = df[config['target']].copy()
    X = df.drop([config['target'], 'latitude(m)', 'longitud(m)'] , axis=1).copy()

    if config['spliting_function']=='random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    elif config['spliting_function']=='spatial':
        X_train, X_test, y_train, y_test = spatial_split(df, config)
    elif config['spliting_function']=='manual':
        train_ds = pd.read_csv("data/train.csv")
        test_ds = pd.read_csv("data/test.csv")
        y_train, X_train, y_test, X_test = train_ds[config["target"]], train_ds.drop(config["target"], axis=1), test_ds[config["target"]], test_ds.drop(config["target"], axis=1)

    # Define the pipeline steps depending on the task (classification or regression)


    if config['model_specific'].lower() =='lightgbm':
        fmin_objective = partial(lgbm_objective, x=X_train,y=y_train)
        ## use null to replace 0
        #geochemical_feat=['Al_1', 'Co_1', 'Cr_1','Cu_1', 'Mg_1', 'Ni_1', 'Pt_1', 'MgOpct_1', 'S_1', 'Ti_1']
        #X_train[geochemical_feat]=X_train[geochemical_feat].replace(0,np.nan)
        #X_test[geochemical_feat]=X_test[geochemical_feat].replace(0,np.nan)

        lgbm_space = {
        'num_leaves': hp.quniform("num_leaves", 8, 80, 4),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'n_estimators':hp.choice("n_estimators", np.linspace(200, 1400, 200, dtype=int))
        }

        best = fmin(fn=fmin_objective, space=lgbm_space, algo=tpe.suggest, max_evals=30)    
        # train the model
        print('>> Training LightGBM...')
        model = lgbm_training(best,X_train,y_train)
        print('>> Done.')
    elif config['model_specific'].lower() == 'randomforest':
        
        cat_feat = list(X_train.loc[:, X_train.dtypes == 'category'].columns.values)
        X_test=X_test.drop(cat_feat, axis=1)
        X_train=X_train.drop(cat_feat, axis=1)
        fmin_objective = partial(rf_objective, x=X_train,y=y_train)
        
        rf_space = {
        'min_samples_split': hp.quniform('min_samples_split', 10, 80, 5),
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'n_estimators':hp.quniform('n_estimators', 100, 1200, 200)
        }
        # drop 
        best = fmin(fn=fmin_objective, space=rf_space, algo=tpe.suggest, max_evals=15)    
        # train the model
        print('>> Training RandomForest...')
        model = rf_training(best,x=X_train,y=y_train)

        print('>> Done.')

    #Show metrics
    try:
        experiment = Experiment(ws, config['experiment'])
        run = experiment.start_logging()
        print('>> Registering models and metrics...')
        run.tag("model", str(config['model_specific']))
        run.tag("spliting_function", str(config['spliting_function']))
        run.tag("metric", str(config['metric']))

        with TemporaryDirectory() as temp_dir:
                joblib.dump(model, temp_dir + '/' + config['model_name'] + '.pkl')
                run.upload_file(config['model_name'] + '.pkl', temp_dir + '/' + config['model_name'] + '.pkl')
                if 'feat_filter' in config:
                    with open(temp_dir + '/' + 'features.json', "w") as file_feat:
                        json.dump(config['feat_filter'], file_feat)
                    run.upload_file('features.json', temp_dir + '/' + 'features.json')
                if 'cate_remove' in config:
                    with open(temp_dir + '/' + 'categories.json', "w") as file_cate: 
                        json.dump(config['cate_remove'], file_cate)
                    run.upload_file('categories.json', temp_dir + '/' + 'categories.json')
                rmtree(temp_dir)
    
        y_pred=model.predict(X_test)

        print('Balanced Accuracy = {}'.format(balanced_accuracy_score(y_test,y_pred)))
        run.log('Balanced Accuracy',balanced_accuracy_score(y_test,y_pred))

        print('F1 Macro = {}'.format(f1_score(y_test,y_pred,average='macro')))
        run.log('F1 Macro',f1_score(y_test,y_pred,average='macro'))

        print('ROC AUC = {}'.format(roc_auc_score(y_test,model.predict_proba(X_test)[::,1])))
        run.log('ROC AUC',roc_auc_score(y_test,model.predict_proba(X_test)[::,1]))

        print('Confusion Matrix\n {}'.format(confusion_matrix(y_test, y_pred)))
        cmtx = { 'schema_type': 'confusion_matrix',
                 'data': {'class_labels': list(model.classes_),
                 'matrix': [[int(y) for y in x] for x in confusion_matrix(y_test, y_pred)]}
                    }
        run.log_confusion_matrix('Confusion Matrix', cmtx)

        if config['create_new_version']=='True' or config['create_new_version']=='true':
            run.register_model(model_name = config['model_name'],
                            model_path = config['model_name'] + '.pkl',
                            datasets =[('Training data', input_ds)],
                            tags = run.get_tags(),
                            description = config['description'] if 'description' in config else '')
        print('>> Done.')

        # Get the most relevant features
        print('>> Generating explainers...')

        explainer = TabularExplainer(model, X_train)
        explanation = explainer.explain_global(X_test)

        # Get an Explanation Client and upload the explanation
        explain_client = ExplanationClient.from_run(run)
        explain_client.upload_model_explanation(explanation)
        print('>> Done.')

        run.complete()
    except Exception as e:
        run.fail(error_details=e)
        print(e)
    
main()
