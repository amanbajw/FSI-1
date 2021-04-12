"""Prepare feature for ML model training. Read from FeatureStore then save to S3 as CSV."""
import sklearn

            
def main():
    insurance_policy_query = insurance_policy_feature_group.athena_query()

    insurance_policy_table = insurance_policy_query.table_name

    query_string = 'SELECT * FROM "'+insurance_policy_table+'"' #+insurance_policy_table
    print('Running ' + query_string)

    # run Athena query. The output is loaded to a Pandas dataframe.
    #dataset = pd.DataFrame()
    insurance_policy_query.run(query_string=query_string, output_location='s3://'+default_s3_bucket_name+'/'+prefix+'/query_results/')
    insurance_policy_query.wait()
    dataset = insurance_policy_query.as_dataframe()

    # Prepare query results for training.
    query_execution = insurance_policy_query.get_query_execution()
    query_result = 's3://'+default_s3_bucket_name+'/'+prefix+'/query_results/'+query_execution['QueryExecution']['QueryExecutionId']+'.csv'
    print(query_result)

    df_features = pd.read_csv(query_result)   
    df_features.columns = feature_names +['PurePremium','Frequency','AvgClaimAmount','eventtime','write_time','api_invocation_time','is_deleted']
    # Select useful columns for training with target column as the first.
    dataset = df_features.iloc[:,np.r_[df_features.columns.get_loc('PurePremium'), 0:60]]

    # Write to csv in S3 without headers and index column.
    dataset.to_csv('dataset.csv', header=False, index=False)
    s3_client.upload_file('dataset.csv', default_s3_bucket_name, prefix+'/training_input/dataset.csv')
    dataset_uri_prefix = 's3://'+default_s3_bucket_name+'/'+prefix+'/training_input/';

    
if __name__ == "__main__":
    main()