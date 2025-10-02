import pandas as pd
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from plots import plot_feature_label_correlations
from regressors import make_linreg_pipeline, make_boosting_pipeline, \
                       make_boosting_nodrop_pipeline
from dataclasses import dataclass

def load_data(path):
    df = pd.read_csv(path, index_col=False)
    # drop index and ID
    df = df.iloc[:, 2:]
    print('Loaded {}. Head:'.format(path))
    print(df.head(5))
    print('Columns: ' + str(df.columns))
    print('df.describe(): ')
    print(df.describe())
    print('Missing values (0 is none missing): ')
    missing = analyze_missing(df)
    print(missing)
    return df

def analyze_missing(df):
    desc = df.describe()
    print(1-desc.loc['count']/len(df))

def drop_missing(df: pd.DataFrame, label):
    len_full = len(df)
    df = df.dropna(subset=[label])
    len_drop = len(df)
    print('Dropped NaNs in {}'.format(label))
    print('Before: {} rows. After: {} rows.'.format(len_full, len_drop))
    return df

def train_hc(df: pd.DataFrame, estimator_cls, seed):
    h_c_y = df.loc[:, 'H/C']
    o_c_y = df.loc[:, 'O/C']
    x = df.loc[:, (df.columns != 'H/C') & (df.columns != 'O/C') & (df.columns != 'O') & (df.columns != 'C')]
    x_tr, x_te, h_c_tr, h_c_te, o_c_tr, o_c_te = train_test_split(x, h_c_y, o_c_y, train_size=0.8, random_state=seed)
    h_model = estimator_cls(seed)
    o_model = estimator_cls(seed)
    h_model.fit(x_tr, h_c_tr)
    o_model.fit(x_tr, o_c_tr)
    h_c_pred = h_model.predict(x_te)
    o_c_pred = o_model.predict(x_te)
    r2_h, r2_o = r2_score(h_c_te, h_c_pred), r2_score(o_c_te, o_c_pred)
    mse_h, mse_o = mean_squared_error(h_c_te, h_c_pred), mean_squared_error(o_c_te, o_c_pred)
    print('R2 H/C {} R2 O/C {}'.format(r2_h, r2_o))
    print('MSE H/C {} MSE O/C {}'.format(mse_h, mse_o))
    return {'h/c': {'r2': r2_h, 'mse': mse_h}, 'o/c': {'r2': r2_o, 'mse': mse_o}}


def train_custom_label(df: pd.DataFrame, estimator_cls, seed, label):
    try:
        y = df.loc[:, label]
    except Exception:
        print('ERROR: df.loc[:, {}] failed.'.format(label))
        exit
    x = df.loc[:, (df.columns != label)]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=0.8, random_state=seed)
    model = estimator_cls(seed)
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    r2 = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    print('R2 {} {}'.format(label, r2))
    print('MSE {} {}'.format(label, mse))
    return {label: {'r2': r2, 'mse': mse}}

def prepare_for_chain_pred(args):
    assert args.chain
    raise NotImplementedError()

def choose_pipeline(model_str):
    if model_str == 'linear':
        return make_linreg_pipeline
    if model_str == 'boosting_nodrop':
        return make_boosting_nodrop_pipeline
    if model_str == 'boosting':
        return make_boosting_pipeline
    raise ValueError('This should never happen')


if __name__ == '__main__':
    parser = ArgumentParser('Basic analysis of biochar data')
    parser.add_argument('path', type=str, help='path to csv file')
    parser.add_argument('--model', choices=['linear', 'boosting_nodrop', 'boosting'], default='linear')
    parser.add_argument('--label', default='NMR aromaticity (%)')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--chain', action='store_true', help='Predict H/C and then aromaticity. \
                                                              Will override arguments')
    parser.add_argument('--repeat', default=1, type=int, help='Bootstrap the model --repeat times')
    parser.add_argument('--plot', action='store_true', help='Plot linear correlations btwn features \
                                                             and args.label')
    # handle args
    args = parser.parse_args()
    if args.chain:
        args = prepare_for_chained_pred(args)
    # prepare model and data
    model_fn = choose_pipeline(args.model)
    df = load_data(args.path)
    df_clean = drop_missing(df, args.label)
    # model fit
    runs = []
    for i in range(args.repeat):
        metrics = train_custom_label(df_clean, make_boosting_pipeline, args.seed+i, args.label)
        runs.append(metrics)
    if args.repeat > 1:
        summary = pd.concat([pd.DataFrame(r).T for r in runs]).groupby(level=0).agg(['mean','std'])
        print(summary)
    # optional plot
    if args.plot:
        plot_feature_label_correlations(df_clean, label)
