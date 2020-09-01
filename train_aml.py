import os
import glob
import shutil
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.data.data_reference import DataReference
from azureml.core import Dataset

ws = Workspace.create(
    "yolo3_ws",
    subscription_id="a6c2a7cc-d67e-4a1a-b765-983f08c0423a",
    resource_group="doorfinder_rg",
    location="westus2",
    exist_ok=True,
)

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

PROJECT_FOLDER = "./aml/"
if os.path.exists(PROJECT_FOLDER):
    shutil.rmtree(PROJECT_FOLDER)

os.makedirs(PROJECT_FOLDER, exist_ok=True)
files = glob.glob("*.py")
for f in files:
    shutil.copy(f, PROJECT_FOLDER)
files = glob.glob("*.cfg")
for f in files:
    shutil.copy(f, PROJECT_FOLDER)
files = glob.glob("*.txt")
for f in files:
    shutil.copy(f, PROJECT_FOLDER)
shutil.copytree("model_data", os.path.join(PROJECT_FOLDER, 'model_data'))
shutil.copytree("yolo3", os.path.join(PROJECT_FOLDER, 'yolo3'))

cd = CondaDependencies.create(pip_packages=['keras==2.1.5', 'tensorflow==1.6.0', 'pillow', 'matplotlib', 'h5py', 'tensorboard'], conda_packages=['python=3.6.11'])
myenv = Environment("yolov3")
myenv.python.conda_dependencies = cd
myenv.python.conda_dependencies.add_pip_package("azureml-sdk")
myenv.python.conda_dependencies.add_channel("conda-forge")
myenv.docker.enabled = True
myenv.docker.base_image = DEFAULT_GPU_IMAGE

# Choose a name for your CPU cluster
CLUSTER_NAME = "gpu-cluster"

# Verify that cluster does not exist already
try:
    aml_cluster = AmlCompute(workspace=ws, name=CLUSTER_NAME)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    print("provisioning new compute target")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_NC6", max_nodes=8, vm_priority="lowpriority"
    )
    aml_cluster = ComputeTarget.create(ws, CLUSTER_NAME, compute_config)

aml_cluster.wait_for_completion(show_output=True)

def_blob_store = ws.get_default_datastore()
# def_blob_store.upload("VOCdevkit", target_path="/data/VOCdevkit")

dataset = Dataset.File.from_files(path=(def_blob_store, '/data/VOCdevkit'))

src = ScriptRunConfig(
    source_directory=PROJECT_FOLDER,
    script='train.py',
    arguments=["--data", dataset.as_named_input('input').as_mount()],
    compute_target=aml_cluster,
    environment=myenv
    )


EXPERIMENT_NAME = "keras-yolo3"

experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)

run = experiment.submit(config=src)
