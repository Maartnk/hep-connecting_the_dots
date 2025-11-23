import os
import shutil
from datetime import datetime


def unique_output_dir(config):
    # Generate a unique directory name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output"]["base_path"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_config_to_output(config_path, output_dir):
    """
    Copies the configuration file to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, output_dir)
