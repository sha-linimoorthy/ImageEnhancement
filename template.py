import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s: %(message)s]')

project_name = "ImageEnhancer"

list_of_files = [
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/train.py",
    f"src/{project_name}/models/",
    f"src/{project_name}/datamodels/",
    "assets/data/data_description.txt",
    "preprocessing/preprocess.py",
    "configs/config.yaml",
    "requirements.txt",
    "main.py",
]

for filePath in list_of_files:
    filePath = Path(filePath)
    fileDir, fileName = os.path.split(filePath)

    if fileDir!="":
        os.makedirs(fileDir,exist_ok = True)
        logging.info(f"Creating directory: {fileDir} for the file: {filePath}")
    if (not os.path.exists(filePath)) or (os.path.getsize(filePath)==0):
        with open(filePath,"w") as file:
            pass
            logging.info(f"Created file: {filePath}")
    else:
        logging.info(f"{filePath} already exits")