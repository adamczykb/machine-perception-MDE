import os
import shutil

from cli.config import ProjectConfig


def ensure_file_not_exist(config: ProjectConfig, file_path: str):
    if os.path.exists(file_path) or os.path.isfile(file_path):
        if (
            config.verbose
            and input(f"File {file_path} exists. Would you like to remove it? [y/n]: ")
            .strip()
            .lower()
            == "y"
        ):
            os.remove(file_path)
            return True
        return False
    return True


def ensure_directory_not_exist(config: ProjectConfig, path: str):
    if os.path.exists(path):
        if not (
            config.verbose
            and input(f"Directory {path} exists. Would you like to remove it? [y/n]: ")
            .strip()
            .lower()
            == "y"
        ):
            raise Exception(f"Cannot extract archive because directory {path} exists")
        shutil.rmtree(path)
