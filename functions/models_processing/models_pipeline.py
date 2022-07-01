from azureml._restclient import snapshots_client
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
import lightgbm as lgbm
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def results_table(automl,task,X_test,y_test,metric):
    board = automl.leaderboard(detailed=True)[['data_preprocessors','balancing_strategy','feature_preprocessors','type']]
    models = []
    accuracy = []
    balanced_accuracy = []
    f1_macro = []
    roc_auc = []
    cnf_mts = []
    class_scores = []
    automl_models = automl.show_models()
    for id in board.index:
        model = Pipeline([('data_preprocessor',automl_models[id]['data_preprocessor']),
                         ('balancing',automl_models[id]['balancing']),
                         ('feature_preprocessor',automl_models[id]['feature_preprocessor']),
                         (task,automl_models[id][task])])
        models.append(model)
        y_hat = model.predict(X_test)
        
        if task=='classifier':
            le = automl.automl_.InputValidator.target_validator
            classes = list(automl.classes_)
            balanced_accuracy_r = balanced_accuracy_score(le.transform(y_test),y_hat)
            balanced_accuracy.append(balanced_accuracy_r)

            f1_macro_r = f1_score(le.transform(y_test),y_hat)
            f1_macro.append(f1_macro_r)

            roc_auc_r = roc_auc_score(le.transform(y_test),y_hat)
            roc_auc.append(roc_auc_r)

            cnf_mtx = confusion_matrix(le.transform(y_test), y_hat)
            cnf_mts.append(cnf_mtx)
            class_score =  dict(zip(classes,cnf_mtx.diagonal()/cnf_mtx.sum(axis=1)))
            class_scores.append(class_score)
        else:
            accuracy_r = accuracy_score(y_test, y_hat)
            accuracy.append(accuracy_r)
    if task=='classifier':
        board['balanced_accuracy'] = balanced_accuracy
        board['f1_macro'] = f1_macro
        board['roc_auc'] = roc_auc
    else:
        board['accuracy'] = accuracy
    board['model'] = models
    if task=='classifier':
        board['confusion_matrix'] = cnf_mts
        board['classes_score'] = class_scores

    board.sort_values(by=[metric],ascending=False)

    return(board)


def lgbm_objective(params, x, y):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'learning_rate': '{:.3f}'.format(params['learning_rate']),
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
    }
    
    clf = lgbm.LGBMClassifier(
        **params
    )
    
    score = cross_val_score(clf, x, y, scoring='neg_log_loss', cv=StratifiedKFold()).mean()
    print("neg_log_loss: {:.3f} params {}".format(score, params))
    return score

def lgbm_training(params, x, y):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
    }
    
    clf = lgbm.LGBMClassifier(
        **params
    )
    clf.fit(x, y)    
    
    return clf


def rf_objective(params, x, y):
    params = {
        'min_samples_split': int(float(params['min_samples_split'])),
        'n_estimators': int(float(params['n_estimators'])),
        'max_depth': int(params['max_depth']),
    }
    
    clf = RandomForestClassifier(**params, n_jobs=-1, criterion='entropy')
    
    score = cross_val_score(clf, x, y, scoring='neg_log_loss', cv=StratifiedKFold()).mean()
    print("neg_log_loss: {:.3f} params {}".format(score, params))
    return score

def rf_training(params, x, y):
    params = {
        'min_samples_split': int(float(params['min_samples_split'])),
        'n_estimators': int(float(params['n_estimators'])),
        'max_depth': int(params['max_depth']),
    }
    
    clf = RandomForestClassifier(**params, n_jobs=-1, criterion='entropy')
    clf.fit(x, y)    
    
    return clf

# Set up azure workspace "DevAIMLWorkspace" using service connection authentication 
def setup_azure_workspace():
    print('>> Setting up Azure client...')
    snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 100000000000

    svc_pr_password = "csO7Q~7iTk4MqlJpY6y_wCWD.~CHZ2kUWKACf"

    svc_pr = ServicePrincipalAuthentication(
        tenant_id="d05d5e5b-385d-4774-b496-d0cf85bfa5f4",
        service_principal_id="50e4bf29-6686-46a5-8632-f07e576725fb",
        service_principal_password=svc_pr_password)


    ws = Workspace(
        subscription_id="755b224d-5144-4359-b618-f62b1efb1c57",
        resource_group="AUAZE-CORP-DEV-EXPLORATIONAI",
        workspace_name="DevAIMLWorkspace",
        auth=svc_pr
        )

    print('>> Done.')
    return ws