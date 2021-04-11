"""Feature engineers the insurance dataset."""
import sklearn
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from sklearn.metrics import mean_absolute_error, mean_squared_error, auc

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_mtpl2(n_samples=100000):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=100000
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True)['data']
    df_freq['IDpol'] = df_freq['IDpol'].astype(np.int)
    df_freq.set_index('IDpol', inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True)['data']

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby('IDpol').sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == np.object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


if __name__ == "__main__":
    df_freq = fetch_openml(data_id=41214, as_frame=True)['data']
    df_freq.head(10)
    df_sev = fetch_openml(data_id=41215, as_frame=True)['data']
    df_sev.head()

    # Loading datasets, basic feature extraction and target definitions
    df = load_mtpl2(n_samples=60000)

    # Note: filter out claims with zero amount, as the severity model
    # requires strictly positive target values.
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Correct for unreasonable observations (that might be data error)
    # and a few exceptionally large claim amounts
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

    df.reset_index(inplace=True)

    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log),
        StandardScaler()
    )

    column_trans = ColumnTransformer(
        [
            ("binned_numeric", KBinsDiscretizer(n_bins=3),["VehAge", "DrivAge"]),
            ("onehot_categorical", OneHotEncoder(), ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            ("passthrough_numeric", "passthrough",["IDpol","VehAge", "DrivAge","VehBrand", "VehPower", "VehGas", "Region", "Area", "ClaimNb","Exposure", "BonusMalus", "ClaimAmount"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,

    )
    X = column_trans.fit_transform(df)

    print(X[0,:])

    bins=[]
    for j,f in enumerate(["VehAge","DrivAge"]):
        for idx, val in enumerate(column_trans.transformers_[0][1].bin_edges_[j]):
            edge1= column_trans.transformers_[0][1].bin_edges_[j][idx-1]
            if idx>0:
                bins.append(f+"_bin"+str(edge1)+"_"+str(val))
    column_trans.transformers_[1][1].get_feature_names(["VehBrand", "VehPower", "VehGas", "Region", "Area"])
    feature_names = bins+\
        column_trans.transformers_[1][1].get_feature_names(["VehBrand", "VehPower", "VehGas", "Region", "Area"]).tolist()+\
        ["Density"]+\
        ["IDpol","VehAge", "DrivAge","VehBrand", "VehPower", "VehGas", "Region", "Area","ClaimNb","Exposure","BonusMalus","ClaimAmount"]


    print(len(feature_names))

    feature_names = [x.replace('.','_') for x in feature_names]
    print(feature_names)

    df_transformed = pd.DataFrame(data=X, columns= feature_names)
    #df_transformed.columns= feature_names
    df_transformed.head()

    df_transformed["PurePremium"] = df_transformed["ClaimAmount"] / df_transformed["Exposure"]
    df_transformed["Frequency"] = df_transformed["ClaimNb"] / df_transformed["Exposure"]
    df_transformed["AvgClaimAmount"] = df_transformed["ClaimAmount"] / np.fmax(df_transformed["ClaimNb"], 1)
    df_transformed[df_transformed.ClaimNb>0].head(20)

    # Save to feature store
