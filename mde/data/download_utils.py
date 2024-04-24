import os
import shutil
import tarfile
import uuid
import requests
import zipfile
from tqdm import tqdm
import subprocess
from cli.config import ProjectConfig
from mde.utils.dialogs import ensure_directory_not_exist, ensure_file_not_exist


def download_archive(config: ProjectConfig) -> None:
    """
    Downloads raw dataset.

    Args:
    - config - YML config file

    Returns:
    - None
    """
    if config.dataset_config == "":
        return

    path = os.path.join(config.dataset_store_achive_path)
    # subprocess.run(f'wget -i {config.dataset_config} -P {path}', shell=True, check=True,stderr=subprocess.STDOUT)
    # with open(config.dataset_config,'r') as files_to_download:
    #     for file_to_download in files_to_download.readlines():
    #         print(file_to_download)
    #         file_ = file_to_download.split('/')[-1]
    #         file_path = os.path.join(
    #             path, f"{file_}"
    #         )

    #         ensure_file_not_exist(config, file_path)
    #         os.makedirs(path, exist_ok=True)

    #         response = requests.get(file_to_download, stream=True, allow_redirects=True)
    #         total_size = int(response.headers.get("content-length", 0))
    #         block_size = 1024  # 1 Kibibyte
    #         print(f"Downloading raw {file_} dataset...")
    #         with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
    #             with open(file_path, "wb") as file:
    #                 for data in response.iter_content(block_size):
    #                     progress_bar.update(len(data))
    #                     file.write(data)


def extract_archive(config: ProjectConfig) -> None:
    """
    Extracts downloaded raw dataset.

    Args:
    - config - YML config file

    Returns:
    - None

    Exceptions:
    - if archive extension format is not supported
    - if there's no archive to extract
    """
    if config.dataset_config == "":
        return

    path = os.path.join(config.dataset_store_achive_path)
    with open(config.dataset_config, "r") as files_to_download:
        for file_to_download in tqdm(files_to_download.readlines()):
            file_ = file_to_download.split("/")[-1].strip()
            print(f"Extracting downloaded {file_} archive...")
            if os.path.exists(path):
                file_path = os.path.join(path, f"{file_}")
                if config.dataset_archive_ext in ["zip"]:
                    try:
                        tmp = os.path.join(
                            f"{config.dataset_directory}/{config.dataset_archive_name}/{file_.split('.')[0]}"
                        )
                        if ensure_file_not_exist(config, tmp):
                            with zipfile.ZipFile(file_path, "r") as zip_ref:
                                zip_ref.extractall(tmp)

                            print(f"Finished extracting {file_}!")
                        
                    except FileNotFoundError:
                        print(f"{file_} does not exist")
                else:
                    raise Exception(
                        f"{config.dataset_archive_ext} format is not supported"
                    )
            else:
                raise Exception(
                    f"Downloaded archive cannot be extracted. Call download_archive function first"
                )
