import pandas as pd
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from plots import plot_feature_label_correlations

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

def drop_missing(df: pd.DataFrame):
    df = df.dropna(subset=['H/C', 'O/C'])
    return df

def train_simple_split(df: pd.DataFrame, estimator_cls, seed):
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
    

def make_linreg_pipeline(seed):
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(with_mean=True, with_std=True),
        LinearRegression(random_state=seed)
    )

def make_boosting_nodrop_pipeline(seed):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        HistGradientBoostingRegressor(random_state=seed)
    )

def make_boosting_pipeline(seed):
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(with_mean=True, with_std=True),
        HistGradientBoostingRegressor(random_state=seed)
    )

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
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    model_fn = choose_pipeline(args.model)
    df = load_data(args.path)
    df_clean = drop_missing(df)
    train_simple_split(df_clean, make_boosting_pipeline, args.seed)
    plot_feature_label_correlations(df_clean, 'H/C')
    plot_feature_label_correlations(df_clean, 'O/C')
