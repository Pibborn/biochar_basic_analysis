from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


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

