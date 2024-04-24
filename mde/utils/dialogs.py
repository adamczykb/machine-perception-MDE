import os
import shutil

from cli.config import ProjectConfig


def ensure_file_not_exist(config: ProjectConfig, file_path: str):
    if os.path.exists(file_path):
        if (
            config.verbose
            and input(f"{file_path} exists. Would you like to remove it? [y/n]: ")
            .strip()
            .lower()
            == "y"
        ):
            shutil.rmtree(file_path)
            return True
        return False
    return True
