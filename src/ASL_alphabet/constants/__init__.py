"""
author:@ kumar dahal
"""
import os
from datetime import datetime
from pathlib import Path

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"



CONFIG_FILE_PATH = Path("configs/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")




##integration testing
BASE_MODEL_PATH_TEST = Path("artifacts/prepare_base_model/base_model.h5")
UPDATED_BASE_MODEL_PATH_TEST = Path("artifacts/prepare_base_model/base_model_updated.h5")
