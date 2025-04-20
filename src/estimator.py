import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import yaml

with open("src/train.yaml", "r") as f:
    training_config = yaml.safe_load(f)

with open("src/configs.yaml", "r") as f:
    network_config = yaml.safe_load(f)


"""
    Configure and execute the SageMaker training job.

    The function performs the following tasks:
    - Reads network credentials and tag settings from the credentials.ini file.
    - Sets up the SageMaker session using the designated S3 bucket.
    - Defines the PyTorch estimator with specified hyperparameters, network configurations,
      checkpoint settings, and metric definitions.
    - Initiates the training job with the provided training and validation directories.
"""

sagemaker_session_bucket = training_config["DATA"]["S3_BUCKET"]

session = sagemaker.Session(
    default_bucket=sagemaker_session_bucket
)

role = sagemaker.get_execution_role()

trainin_dir = training_config["DATA"]["S3_TRAIN_DIR"]
val_dir = training_config["DATA"]["S3_VAL_DIR"]

timestamp = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%Y%m%d_%H-%M-%S')

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g5.8xlarge",
    instance_count=1,
    role=role,
    pytorch_version="2.2",
    framework_version="2.2",
    py_version="py310",
    hyperparameters={
        "model_name": training_config["PROJECT"]["PROJECT_NAME"],
        "epochs": training_config["TRAINING"]["EPOCHS"],
        "batch_size": training_config["TRAINING"]["BATCH_SIZE"],
        "learning_rate": training_config["TRAINING"]["LEARNING_RATE"],
        "num_classes": training_config["MODEL"]["NUM_CLASSES"],
        "resize": training_config["MODEL"]["RESIZE"],
        "timestamp": timestamp
    },
    base_job_name=f"{training_config['PROJECT']['PROJECT_NAME']}-RGB",
    tags=[
        {"Key": "Application", "Value": network_config['TAGS']['application']},
        {"Key": "Cost Center", "Value": network_config['TAGS']['cost_center']}
    ], 
    subnets=network_config['NETWORK']['subnets'].split(','),
    security_group_ids=network_config['NETWORK']['security_group_ids'].split(','),
    checkpoint_s3_uri=os.path.join(training_config["DATA"]["S3_CHECKPOINT_DIR"], f"{timestamp}"),
    checkpoint_local_path="/opt/ml/checkpoints",
    output_path=training_config["DATA"]["S3_OUTPUT_DIR"] + "/" + timestamp,
    environment={
        "WANDB_API_KEY": network_config["WANDB"]["wandb_api_key"],
        "WANDB_MODE": "offline",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    metric_definitions=[
        {'Name': 'epoch', 'Regex': "EPOCH: ([0-9\\.]+)"},
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'train_accuracy: ([0-9\\.]+)'},
        {'Name': 'val:loss', 'Regex':  'val_loss: ([0-9\\.]+)'},
        {'Name': 'val:accuracy', 'Regex':  'val_accuracy: ([0-9\\.]+)'},
        {'Name': 'val:f1_score', 'Regex':  'f1_score: ([0-9\\.]+)'},
    ],
    enable_sagemaker_metrics=True,
    requirements_file="requirements.txt"
)

estimator.fit(
    inputs={
        "train": trainin_dir,
        "val": val_dir
    }
)
