"""Feature engineers the insurance dataset and save result into feature store"""
import sklearn
from functools import partial

import logging

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
import boto3
import sagemaker

from sagemaker.session import Session
from sagemaker import get_execution_role
from time import gmtime, strftime, sleep
from sagemaker.feature_store.feature_group import FeatureGroup
import time

# You can modify the following to use a role of your choosing. See the documentation for how to create this.
role = get_execution_role()
region = boto3.Session().region_name

boto_session = boto3.Session(region_name=region)

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)
s3_client = boto3.client('s3', region_name=region)
account_id = boto3.client('sts').get_caller_identity()["Account"]

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

def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label].name in ['object','category']:
            data_frame[label] = data_frame[label].astype("str").astype("string")
def wait_for_feature_group_creation_complete(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")

def feature_engineering():
    # Feature Engineering
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

    df_transformed["PurePremium"] = df_transformed["ClaimAmount"] / df_transformed["Exposure"]
    df_transformed["Frequency"] = df_transformed["ClaimNb"] / df_transformed["Exposure"]
    df_transformed["AvgClaimAmount"] = df_transformed["ClaimAmount"] / np.fmax(df_transformed["ClaimNb"], 1)
    df_transformed[df_transformed.ClaimNb>0].head(20)
    return df_transformed

def save_to_feature_store(df_transformed):
    df_data = df_transformed.copy()
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )    
    # Setup S3 offline feature store
    # You can modify the following to use a bucket of your choosing
    default_s3_bucket_name = feature_store_session.default_bucket()
    prefix = 'sagemaker-featurestore-insurance'
    print(default_s3_bucket_name)
            
    # Note: In this example we use the default SageMaker role, assuming it has both AmazonSageMakerFullAccess and AmazonSageMakerFeatureStoreAccess managed policies. If not, please make sure to attach them to the role before proceeding. 
    # Define feature group
    insurance_policy_feature_group_name = 'insurance-policy-feature-group-' + strftime('%d-%H-%M-%S', gmtime())
    print(insurance_policy_feature_group_name)

    insurance_policy_feature_group = FeatureGroup(name=insurance_policy_feature_group_name, sagemaker_session=feature_store_session)
    print(insurance_policy_feature_group)

    current_time_sec = int(round(time.time()))

    # cast object dtype to string. The SageMaker FeatureStore Python SDK will then map the string dtype to String feature type.
    cast_object_to_string(df_data)
    # record identifier and event time feature names
    record_identifier_feature_name = "IDpol"
    event_time_feature_name = "EventTime"

    # append EventTime feature
    df_data[event_time_feature_name] = pd.Series([current_time_sec]*len(df_data), dtype="float64")
    # load feature definitions to the feature group. SageMaker FeatureStore Python SDK will auto-detect the data schema based on input data.
    insurance_policy_feature_group.load_feature_definitions(data_frame=df_data); # output is suppressed
    insurance_policy_feature_group.create(
        s3_uri=f"s3://{default_s3_bucket_name}/{prefix}",
        record_identifier_name=record_identifier_feature_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=role,
        enable_online_store=True
    )

    wait_for_feature_group_creation_complete(feature_group=insurance_policy_feature_group)
    insurance_policy_feature_group.describe()  
    # ingest data into feature store
    insurance_policy_feature_group.ingest(
        data_frame=df_data, max_workers=5, wait=True
    )


    '''
    insurance_policy_feature_group_s3_prefix = prefix + '/' + account_id + '/sagemaker/' + region + '/offline-store/' + insurance_policy_feature_group_name + '/data'

    offline_store_contents = None
    while (offline_store_contents is None):
        objects_in_bucket = s3_client.list_objects(Bucket=default_s3_bucket_name,Prefix=insurance_policy_feature_group_s3_prefix)
        if ('Contents' in objects_in_bucket and len(objects_in_bucket['Contents']) > 1):
            offline_store_contents = objects_in_bucket['Contents']
        else:
            print('Waiting for data in offline store...\n')
            sleep(60)

    print('Data available.')'''
            
def main():
    df = feature_engineering()
    save_to_feature_store(df)

if __name__ == "__main__":
    main()