"""Feature engineers the insurance dataset and save result into feature store"""
import subprocess, sys

from functools import partial

import logging, argparse

import numpy as np
import pandas as pd

import boto3
import sagemaker

from sagemaker.session import Session
from sagemaker import get_execution_role
from time import gmtime, strftime, sleep
from sagemaker.feature_store.feature_group import FeatureGroup
import time
from time import gmtime, strftime, sleep

original_version = sagemaker.__version__
if sagemaker.__version__ != "2.20.0":
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "sagemaker==2.20.0"]
    )
    import importlib
    importlib.reload(sagemaker)

    
region = boto3.Session().region_name
default_bucket = sagemaker.session.Session().default_bucket()
prefix = 'sagemaker-featurestore-insurance'

boto_session = boto3.Session(region_name=region)

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)
s3_client = boto3.client('s3', region_name=region)
account_id = boto3.client('sts').get_caller_identity()["Account"]

feature_group_name='insurance-policy-feature-group-13-01-12-16'
feature_group = None

feature_s3_url = None

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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

'''
Check if given Feature Group name exists
'''
def feature_group_exist(group_name):
    groups = sagemaker_client.list_feature_groups()
    if groups is not None and groups.get("FeatureGroupSummaries") is not None and len(groups["FeatureGroupSummaries"]) > 0:
        for group in groups["FeatureGroupSummaries"]:
            if group["FeatureGroupName"] == group_name:
                return True
    return False


def save_to_feature_store():
    logger.info("Save to FeatureStore started")
    global feature_group
    
    df_data = pd.read_csv(feature_s3_url)
    logger.info("Read data from S3: %s", df_data.head())
    
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )    
    # You can modify the following to use a bucket of your choosing
    logger.info("Default bucket: %s", default_bucket)
                
    # record identifier and event time feature names
    record_identifier_feature_name = "IDpol"
    event_time_feature_name = "EventTime"
    current_time_sec = int(round(time.time()))
    # cast object dtype to string. The SageMaker FeatureStore Python SDK will then map the string dtype to String feature type.
    cast_object_to_string(df_data)
    df_data[event_time_feature_name] = pd.Series([current_time_sec]*len(df_data), dtype="float64")

    feature_group_name = 'insurance-policy-feature-group-' + strftime('%d-%H-%M-%S', gmtime())
    logger.info("Feature Group Name: %s", feature_group_name)

    # Check if feature group already exists. Create a feature group if doesn't exist.
    if feature_group_exist(feature_group_name) == False:
        logger.info("Feature Group: %s doesn't exist. Create a new one.", feature_group)

        feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)

        # append EventTime feature
        # load feature definitions to the feature group. SageMaker FeatureStore Python SDK will auto-detect the data schema based on input data.
        feature_group.load_feature_definitions(data_frame=df_data); # output is suppressed
        feature_group.create(
            s3_uri=f"s3://{default_bucket}/{prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=get_execution_role(),
            enable_online_store=True
        )

        wait_for_feature_group_creation_complete(feature_group=feature_group)
        feature_group.describe()  
    else:
        logger.info("Feature Group: %s exits", feature_group)
        # Init feature group object if already exists
        feature_group = FeatureGroup(
            name=feature_group_name, 
            sagemaker_session=feature_store_session)

    # ingest data into feature store
    feature_group.ingest(
        data_frame=df_data, max_workers=5, wait=True
    )


def prepare_input():
    logger.info("Start preparing ML input")
    global feature_group
    global feature_names
    
    insurance_policy_query = feature_group.athena_query()

    insurance_policy_table = insurance_policy_query.table_name

    query_string = 'SELECT * FROM "'+insurance_policy_table+'"' #+insurance_policy_table
    logger.info("Running: %s", query_string)

    # run Athena query. The output is loaded to a Pandas dataframe.
    #dataset = pd.DataFrame()
    insurance_policy_query.run(query_string=query_string, output_location='s3://'+default_bucket+'/'+prefix+'/query_results/')
    insurance_policy_query.wait()
    dataset = insurance_policy_query.as_dataframe()

    # Prepare query results for training.
    query_execution = insurance_policy_query.get_query_execution()
    query_result = 's3://'+default_bucket+'/'+prefix+'/query_results/'+query_execution['QueryExecution']['QueryExecutionId']+'.csv'
    logger.info("Query result: %s", query_result)
    
    df_features = pd.read_csv(query_result)
    df_features.columns = feature_names +['PurePremium','Frequency','AvgClaimAmount','eventtime','write_time','api_invocation_time','is_deleted']
    # Select useful columns for training with target column as the first.
    dataset = df_features.iloc[:,np.r_[df_features.columns.get_loc('PurePremium'), 0:60]]

    pd.DataFrame(dataset).to_csv("/opt/ml/processing/training_input/dataset.csv", mode='w+', header=False, index=False)

    
def main():
    save_to_feature_store()
    prepare_input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_s3_url", type=str, required=True)
    parser.add_argument("--feature_group_name", type=str, requred=False)
    args = parser.parse_args()
 
    feature_s3_url = args.feature_s3_url
    if args.feature_group_name is not None:
        feature_group_name = args.feature_group_name

    logger.info("FeatureStore arguments: ", feature_s3_url)
    
    main()
    logger.info("FeatureStore completed")
