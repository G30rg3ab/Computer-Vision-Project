import torch
import boto3
import os

def load_model(self, checkpoint_path):
    """
    Loads a saved model checkpoint into the model.
    Supports loading from local storage and S3.

    Args:
    - checkpoint_path (str): Local file path or S3 URI to the model checkpoint.

    Returns:
    - None: Updates the model in-place.
    """
    if checkpoint_path.startswith("s3://"):
        print(f"=> Downloading model from {checkpoint_path}...")

        # Extract S3 bucket and key
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = checkpoint_path.split("/")[2]
        s3_key = "/".join(checkpoint_path.split("/")[3:])

        # Temporary local file path
        local_checkpoint = "temp_model.pth"
        
        # Download model from S3
        try:
            s3_client.download_file(bucket_name, s3_key, local_checkpoint)
            checkpoint_path = local_checkpoint  # Update path to local file
            print(f"=> Model downloaded from S3 to {local_checkpoint}")
        except Exception as e:
            print(f" Failed to download model from S3: {e}")
            return

    print(f"=> Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
    # Load model state dict
    self.model.load_state_dict(checkpoint["state_dict"])
    
    print("=> Model successfully loaded!")

    # Remove temp file if downloaded from S3
    if checkpoint_path == "temp_model.pth":
        os.remove(checkpoint_path)



def upload_s3(local_file_path, s3_bucket, s3_key):
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        print(f"File uploaded to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload file to S3: {e}")