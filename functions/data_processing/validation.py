import pandas as pd
from sys import exit
from cerberus import Validator

def validate_config(config,pipeline_name,df):
    if pipeline_name == 'prepro_train':
        try:
            local = list(df[config['local_explainer']['feature']].unique())
        except:
            local = []
        schema = {
                    "ds_name" :{'type':'string'}, 
                    "ds_version" :{'type':'string'},
                    "experiment" :{'type':'string'},
                    "model_name" :{'type':'string'},
                    "create_new_version" :{'type':'string','allowed': ['true','True','false','False']},
                    "target" :{'type':'string', 'allowed': ['group_rocktype', 'eon_p', 'era_p', 'gravont_1', 'gravont_2', 'gravont_3', 'magont_1_1', 'magont_1_2', 'magont_1_3', 'ongravity_1', 'ongravity_2', 'ongravity_3', 'ongrv1vd_1', 'ongrv1vd_2', 'ongrv1vd_3', 'onmago1vd_1', 'onmago1vd_2', 'onmagonl_1', 'onmagonl_2', 'onmagonl_3', 'engdesc', 'dist_2_crust_thick', 'seis_moho', 'fault_type', 'geoline_type', 'dist_2_geolines', 'dist_2_mum_sed', 'mafic', 'lon', 'lat', 'onmago1vd_3', 'dist_2_faults', 'longitud(m)', 'latitude(m)']},
                    "spliting_function" :{'type':'string', 'allowed': ['random','spatial','manual']},
                    "feat_filter" :{'required': False,'type': 'dict', 'schema' :{
                                                                'method': {'required': False, 'type':'string','allowed':{'select','remove'},'required': False}, 
                                                                'features':{'required': False, 'type':'list','allowed': ['group_rocktype', 'eon_p', 'era_p', 'gravont_1', 'gravont_2', 'gravont_3', 'magont_1_1', 'magont_1_2', 'magont_1_3', 'ongravity_1', 'ongravity_2', 'ongravity_3', 'ongrv1vd_1', 'ongrv1vd_2', 'ongrv1vd_3', 'onmago1vd_1', 'onmago1vd_2', 'onmagonl_1', 'onmagonl_2', 'onmagonl_3', 'engdesc', 'dist_2_crust_thick', 'seis_moho', 'fault_type', 'geoline_type', 'dist_2_geolines', 'dist_2_mum_sed', 'mafic', 'lon', 'lat', 'onmago1vd_3', 'dist_2_faults', 'longitud(m)', 'latitude(m)'],'dependencies':['method']}
                                                                }},
                    "cate_remove" :{'required': False,'type': 'dict', 'schema' :{
                                                                'method': {'required': False, 'type':'string','allowed':{'delete','replace'}},
                                                                'rep_name': {'required': False, 'type':'string','dependencies':{'method':['replace']}},
                                                                'categories':{'required': False, 'type':'list','allowed': list(df[config['target']].unique()),'dependencies':['method']}
                                                                }},                                                
                    "time_left_for_this_task" :{'type':'integer', 'min': 60},
                    "metric" :{'type':'string', 'allowed':  ['accuracy', 'balanced_accuracy', 'f1_macro', 'auc']},
                    "model_specific" :{'type':'string','allowed': ['True', 'true', 'lightgbm','randomforest']},
                    "local_explainer" :{'required': False,'type':'dict', 'schema' :{
                                                                'feature':{'required': False, 'type':'string','allowed': list(df.columns)},
                                                                'category':{'required': False, 'type':'string','allowed':  local,'dependencies':['feature']},
                                                                }},
                    "tags" :{'required': False, 'type':'string'},     
                    "description" :{'required': False, 'type':'string'},
                    "interpretable":{'required': False, 'type':'integer', 'min': 1, 'max':2}
                }
    else:
        schema = {

                }
    v = Validator(schema)
    value = v.validate(config)
    if value  == False:
        print(v.errors)
        exit()
    return value

def check_version(container,path,file_name,create_new_version):
    if not file_name:
        raise ValueError("File name cannot be none.")
    version = 1
    while(True):
        if not(container.get_blob_client(path + 'Tabular\\Version ' + str(version)+'\\'+ file_name + '.csv').exists()):
            break
        version = version + 1    
    
    if version== 1 or create_new_version=="True" or create_new_version=="true":
        file_name_out = 'Version ' + str(version)+'\\'+ file_name
        return(file_name_out)
    else:
        file_name_out = 'Version ' + str(version-1)+'\\'+ file_name
        return(file_name_out)

def check_extraction_methods(extraction_methods):
    
    if not extraction_methods:
        raise ValueError("Extraction method cannot be none.")

    for extraction_method in extraction_methods:
        if not (extraction_method in ['input_points','position','distance']):
            raise ValueError("Invalid extraction method: {0}".format(extraction_method))

def check_input_points(input_points_names):
    
    if not input_points_names:
        raise ValueError("Input Points cannot be none.")
    elif not len(input_points_names)==1:
        raise ValueError("More than one input points.")

def check_features(features ,file_name, file_features):

    if features=='/*NA*/':
        raise ValueError("Feature cannot be none.")
    try: 
        temp_feat = features.copy().remove('distance')
    except:
        temp_feat = features.copy()
    if temp_feat and not set(temp_feat).issubset(file_features):
        raise ValueError("Invalid features for {0}".format(file_name))

def check_format(df):
    for c in df.columns:
        df_temp = df[c].copy()
        df_temp = pd.to_numeric(df_temp, errors='coerce')
        if df_temp.isnull().sum() < len(df_temp):
            df[c] = df_temp
    return df

def check_if_blob_exists(container, blob_names):

    if not blob_names:
        raise ValueError("Block blob names cannot be none.")

    for blob_name in blob_names:
        if not container.get_blob_client(blob_name).exists():
            raise ValueError("File '{0}' NOT found!".format(blob_name))