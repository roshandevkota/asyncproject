import pandas as pd
import numpy as np
from collections import Counter
import os
import json
import pickle
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import optuna
from optuna.integration import CatBoostPruningCallback

from collections import defaultdict





def get_meta(df_original, target_column, force_col_types=None):
    df_copy = df_original.copy()
    if target_column in df_original.columns:
        df_copy.rename(columns={target_column: 'target'}, inplace=True)

    data, meta = get_data(df_copy, force_col_types=force_col_types)
    return data, meta

def get_data(df, max_str=0.2, force_col_types=None):
    if force_col_types is None:
        force_col_types = {}

    def count_elements(input_list):
        element_counts = Counter(input_list)
        return element_counts

    def remove_nan_cell(col):
        col = [value for value in col if pd.notna(value) and value != '']
        return col

    def identify_numerical(col):
        num_conv = [str2float(c) for c in col]
        num_count = [1 for x in col if isinstance(x, str)]
        return num_conv, num_count, sum(num_count) / len(col)

    def median_pickone(x):
        x_sorted = sorted(x)
        return x_sorted[len(x)//2]

    def make_conv(num_conv, max_str=0.2):
        cleaned_col = remove_nan_cell(num_conv)
        num_str = sum([isinstance(n, str) * 1 for n in cleaned_col])
        num_frac = num_str / len(cleaned_col)
        is_num = num_frac < max_str

        if is_num:
            fill_value = np.nan
            res_list = [fill_value if isinstance(n, str) else n for n in num_conv]
        else:
            res_list = [str(n) for n in num_conv]

        return res_list, is_num

    def detect_categorical(col, min_cat_frac=0.4):
        cat_frac = len(np.unique(col)) / len(col)
        return cat_frac < min_cat_frac

    def mode(x):
        vals, cnt = np.unique(x[~np.isnan(x)], return_counts=True)
        i = np.argmax(cnt)
        return vals[i]

    def fill_cat(col):
        return mode(np.asarray(col))

    def fill_num(col):
        return np.nanmean(np.asarray(col))

    def fill_nan(col, fill_method):
        fill_val = fill_method(col)
        return np.nan_to_num(col, nan=fill_val), fill_val

    def cnt_list(lst, x):
        return sum([1 for l in lst if l == x])

    def convert_str(col):
        non_nan_values = [val for val in col if not pd.isna(val)]
        non_nan_values = sorted(set(non_nan_values))
        cnt_vals = [(cnt_list(col, v), v) for v in non_nan_values]
        a, i = sorted(cnt_vals, key=lambda x: x[0])[-1]
        col = np.where(pd.isna(col), i, col)

        vals = sorted(set(col))
        conv_dict = {v: i for i, v in enumerate(vals)}
        cnt_vals = [(cnt_list(col, v), v) for v in vals]
        max_c, max_v = sorted(cnt_vals, key=lambda x: x[0])[-1]

        return np.asarray([conv_dict[c] for c in col]), conv_dict, max_v

    df_res = pd.DataFrame(columns=df.columns)
    meta_info = {}

    for col in df.columns:
        c = df[col].values
        col_str, num_count, num_frac = identify_numerical(c)
        col_conv, is_num = make_conv(col_str, max_str=max_str)
        forced_type = force_col_types.get(col, None)

        if is_num:
            col_conv = np.asarray(col_conv)
            detected_as_cat = detect_categorical(col_conv)
            if forced_type:
                if forced_type == 'cat':
                    detected_as_cat = True
                elif forced_type == 'non_cat':
                    detected_as_cat = False

            if detected_as_cat:
                is_cat = True
                res_c, fill_val = fill_nan(col_conv, fill_cat)
                meta_info[col] = {'known': set(res_c)}
            else:
                is_cat = False
                res_c, fill_val = fill_nan(col_conv, fill_num)
                meta_info[col] = {}
            meta_info[col].update({'is_cat': is_cat, 'fill': fill_val})
        else:
            res_c, conv_dict, max_v = convert_str(c)
            if forced_type == 'cat':
                meta_info[col] = {'is_cat': True, 'fill': max_v, 'conv': conv_dict}
            elif forced_type == 'non_cat':
                meta_info[col] = {'is_cat': False, 'fill': max_v, 'conv': conv_dict}
            else:
                meta_info[col] = {'is_cat': True, 'fill': max_v, 'conv': conv_dict}

        df_res[col] = res_c

    return df_res, meta_info

def str2float(x):
    try:
        return float(x)
    except ValueError:
        return x




def convert_lists_to_sets(obj):
    if isinstance(obj, list):
        return set(obj)
    if isinstance(obj, dict):
        return {key: convert_lists_to_sets(value) for key, value in obj.items()}
    if isinstance(obj, set):
        return {convert_lists_to_sets(item) for item in obj}
    return obj

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    return obj






def train_with_parameter_tuning_test(df_original, forced_types, target_column, num_iterations, csv_file_name):
    target_type = None
    model_type = None
    performance_score = None

    data, meta = get_meta(df_original, target_column, forced_types)
    is_target_categorical = meta['target']['is_cat']

    train_pool, test_pool, X_train, X_test, y_train, y_test, cat_features = split_train_test_pool(data, meta)
    print("meta",meta)
    print("X_train",X_train)
    print("X_test",X_test)
    print("y_train",y_train)
    print("y_test",y_test)
    print("cat_features",cat_features)
    print("num_iterations",num_iterations)

    if is_target_categorical:
        target_type = 'Categorical'
        model_type = 'Classifier'
    else:
        target_type = 'Not Categorical'
        model_type = 'Regression'

    study = optuna.create_study(direction="maximize" if is_target_categorical else "minimize")
    print("Starting optimization")
    study.optimize(lambda trial: Parameter_tuning(trial, is_target_categorical, train_pool, test_pool), n_trials=num_iterations, timeout=600)

    best_params = study.best_params
    best_model = study.best_trial.user_attrs['model']

    models_directory = os.path.join(settings.MEDIA_ROOT, 'models')
    os.makedirs(models_directory, exist_ok=True)

    base_file_name = os.path.basename(csv_file_name)
    model_name = f"{os.path.splitext(base_file_name)[0]}_trained_model.pkl"
    model_path = os.path.join(models_directory, model_name)
    meta_name = f"{os.path.splitext(base_file_name)[0]}_meta.json"
    meta_path = os.path.join(models_directory, meta_name)

    with open(model_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    meta = convert_sets_to_lists(meta)

    # with open(meta_path, 'w') as meta_file:
    #     json.dump(meta, meta_file)

    performance_score = calculate_model_performance(best_model, is_target_categorical, test_pool)

    return "Model trained successfully", target_type, model_type, performance_score, model_path, meta_path








def split_train_test_pool(data,meta):
    X = data.drop('target', axis=1)
    y = data['target']
    X = X.applymap(lambda x: str(x) if isinstance(x, float) else x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    cat_features = [col for col, info in meta.items() if info['is_cat'] and col != 'target']
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(X_test, label=y_test, cat_features=cat_features)
    
    return train_pool,test_pool,X_train, X_test, y_train, y_test, cat_features



def Parameter_tuning(trial, is_target_categorical, train_pool, test_pool):
    print("Training with parameters: ", trial)

    # train_pool, test_pool, X_train, X_test, y_train, y_test, cat_features = split_train_test_pool(data, meta)

    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "loss_function": 'MultiClass' if is_target_categorical else 'RMSE',
        "eval_metric": 'Accuracy' if is_target_categorical else 'RMSE'
    }

    model = CatBoostClassifier(**param) if is_target_categorical else CatBoostRegressor(**param)


    pruning_callback = CatBoostPruningCallback(trial, "Accuracy" if is_target_categorical else "RMSE")


    model.fit(train_pool, eval_set=test_pool, verbose=False, early_stopping_rounds=100, callbacks=[pruning_callback])


    performance_score = calculate_model_performance(model, is_target_categorical, test_pool)


    trial.set_user_attr('model',model)

    return performance_score


def calculate_model_performance(model, is_target_categorical, test_pool):
    y_true = test_pool.get_label()

    predictions = model.predict(test_pool)
    text=None

    if is_target_categorical:
        performance_score = accuracy_score(y_true, predictions)
        print("Accuracy on the test set: ", performance_score)
        text = 'Accuracy on the test set: '+ str( performance_score)
    else:
        performance_score = r2_score(y_true, predictions)
        print("R-squared (R2) score on the test set: ", performance_score)
        text = 'R-squared (R2) score on the test set: '+ str(performance_score)

    return performance_score





def compare_metadata(new_metadata, metadata_path):
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    old_metadata = convert_lists_to_sets(old_metadata)
    return new_metadata == old_metadata



















def trans_data(df, meta, reverse=False):
    conv_columns = df.columns
    df_res = pd.DataFrame(columns=conv_columns)
    for col in meta:
        if col not in conv_columns:
            continue
        c = df[col]
        c_str2float = [str2float(cc) for cc in c.values]
        if meta[col]['is_cat']:
            if 'conv' in meta[col]:
                conv_dc = defaultdict(lambda: meta[col]['conv'].get(meta[col]['fill'], None))
                conv_dc.update(meta[col]['conv'])
                if reverse:
                    conv_dc = {v: k for k, v in conv_dc.items()}
                df_res[col] = [conv_dc[cc] for cc in c]
            else:
                values = meta[col]['known']
                fill_value = meta[col]['fill']
                res_list = [cc if cc in values else fill_value for cc in c_str2float]
                df_res[col] = res_list
        else:
            fill_value = meta[col]['fill']
            num_conv = [cc for cc in c_str2float]
            res_list = [fill_value if isinstance(n, str) else n for n in num_conv]
            df_res[col] = res_list
    return df_res

def make_prediction(input_df, model_path, meta_path):
    with open(meta_path, 'r') as file:
        loaded_meta = json.load(file)

    input_df = trans_data(input_df, loaded_meta)
    input_df = input_df.applymap(lambda x: str(x) if isinstance(x, float) else x)

    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    if 'target' in input_df.columns:
        input_df = input_df.drop('target', axis=1)
    predictions = loaded_model.predict(input_df)
    print("predictions",predictions)
    input_df['target'] = predictions

    predictionsdf = input_df
    print("predictionsdf",predictionsdf)
    final_df = trans_data(predictionsdf, loaded_meta, True)
    print("final_df",final_df)
    return final_df
