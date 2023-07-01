import os

# Cloud SDK Import Statements 
import boto3
import azure.storage
import google.cloud

# Cloud SDK Global Variables
aws_access_key_id = os.environ['aws_access_key_id']
aws_secret_access_key = os.environ['aws_secret_access_key']
azure_storage_name = os.environ['azure_storage_name']
azure_storage_key = os.environ['azure_storage_key']
google_project_id = os.environ['google_project_id']

# Boto3 
# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
s3_client = boto3.client('s3',
          aws_access_key_id=aws_access_key_id,
          aws_secret_access_key=aws_secret_access_key
          )
         
# Azure 
# Azure Storage SDK for Python allows you to access storage using the Azure Storage services.
azure_blob_client = azure.storage.BlobServiceClient(
    account_name=azure_storage_name,
    account_key=azure_storage_key
)

# Google Cloud 
# Google Cloud SDK is the official repository for Google Cloud tools and frameworks that enable access to services.
gcp_client = google.cloud.storage.Client(
    project=google_project_id
)
