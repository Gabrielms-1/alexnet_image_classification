import os
import boto3
import subprocess

"""
Module for synchronizing wandb checkpoints between S3 and the local directory.

This module downloads wandb checkpoints from an S3 bucket to a local directory and synchronizes
local checkpoints containing the word "offline" to the wandb server.
"""

s3_bucket = "cad-brbh-datascience"
s3_prefix = "alexnet_image_classification/checkpoints/wandb"
local_dir = "wandb"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket)

def sync_wandb_from_s3(obj):
    """
    Downloads a wandb checkpoint file from S3 to the local directory.

    Parameters:
    obj (boto3.s3.ObjectSummary): The S3 object that will be downloaded.
    """
    target = os.path.join(local_dir, os.path.relpath(obj.key, s3_prefix))
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    
    bucket.download_file(obj.key, target)

def sync_wandb_to_s3(checkpoint):
    """
    Synchronizes a local checkpoint to the wandb remote if it is offline.

    Parameters:
    checkpoint (str): The name of the checkpoint file in the local directory.
    """
    if "offline" in checkpoint:
        print(f"Syncing {checkpoint} to wandb remote")
        subprocess.run(["wandb", "sync", os.path.join(local_dir, checkpoint)])

if __name__ == "__main__":
    print("Syncing wandb from s3 to local")
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        sync_wandb_from_s3(obj)
    
    print("Syncing local wandb to s3")    
    for checkpoint in os.listdir(local_dir):
        sync_wandb_to_s3(checkpoint)
    
    print("Syncing complete")
    

