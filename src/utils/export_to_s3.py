import os
import boto3
import concurrent.futures

def upload_directory_to_s3(directory, bucket, s3_prefix):
    s3 = boto3.client('s3')
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for root, dirs, files in os.walk(directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, directory)
                s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')
                tasks.append(executor.submit(s3.upload_file, local_path, bucket, s3_key))
        concurrent.futures.wait(tasks)

if __name__ == '__main__':
    local_directory = 'data/full'
    s3_bucket = 'cad-brbh-datascience'
    s3_directory = 'alexnet_image_classification/data/full'
    upload_directory_to_s3(local_directory, s3_bucket, s3_directory)
