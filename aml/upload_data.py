"""
Use this script to upload your data to Azure Blob Storage. 
You should generally run this script before running `train_aml.py`.
"""

import json
from azureml.core import Workspace

with open("aml/config.json", "r") as f:
    config = json.load(f)

ws = Workspace.create(
    config["workspace_name"],
    subscription_id=config["subscription_id"],
    resource_group=config["resource_group"],
    location=config["location"],
    exist_ok=True,
)

def_blob_store = ws.get_default_datastore()
def_blob_store.upload("VOCdevkit", target_path="/data/VOCdevkit")
