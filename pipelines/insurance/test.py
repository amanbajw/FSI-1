import boto3
import sagemaker


region = boto3.Session().region_name
role = sagemaker.get_execution_role()
default_bucket = sagemaker.session.Session().default_bucket()

# Change these to reflect your project/business name or if you want to separate ModelPackageGroup/Pipeline from the rest of your team
model_package_group_name = f"sagemaker-group-insurance"
pipeline_name = f"sagemaker-pipeline-insurance"
print(role)

from pipeline import get_pipeline


pipeline = get_pipeline(
    region=region,
    role=role,
    default_bucket=default_bucket,
    model_package_group_name=model_package_group_name,
    pipeline_name=pipeline_name,
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()