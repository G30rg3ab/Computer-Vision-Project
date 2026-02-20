import torch
import boto3
import os
## functions to upload/download files from amazon s3 ##

def download_from_s3(s3_uri):
    if s3_uri.startswith("s3://"):
        
        uri_parts = s3_uri.split("/")
        s3_bucket = uri_parts[2]
        s3_object_key = "/".join(uri_parts[3:])
        
        local_file_path = os.path.basename(s3_object_key)
        
        s3_client = boto3.client("s3", region_name="us-east-1")
        try:
            s3_client.download_file(s3_bucket, s3_object_key, local_file_path)
            s3_uri = local_file_path  
            print(f"=> Model downloaded from S3 to {local_file_path}")
        except Exception as e:
            print(f"No files downloaded from s3")
            return None


def upload_file_to_s3(local_file_path, s3_target_dir):
    """
    Upload a local file to an S3 target directory, keeping its local filename the same.
    """
    if not s3_target_dir.startswith("s3://"):
        print("Target path is not an S3 URI. No upload performed.")
        return None
    print(f"=> Uploading file to {s3_target_dir}...")

    # Extract bucket and prefix from the S3 URI
    parts = s3_target_dir.split("/")
    bucket_name = parts[2]
    prefix = "/".join(parts[3:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Get the local filename and build the S3 key
    filename = os.path.basename(local_file_path)
    s3_key = prefix + filename
    print(f"Bucket: {bucket_name}")
    print(f"S3 key: {s3_key}")

    # Create the S3 client (adjust region if necessary)
    s3_client = boto3.client("s3", region_name="us-east-1")
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        full_s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"=> File uploaded to {full_s3_uri}")
        return full_s3_uri
    except Exception as e:
        print(f"Failed to upload file to S3: {e}")
        return None
    

