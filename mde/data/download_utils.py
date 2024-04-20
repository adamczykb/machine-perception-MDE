import os
import shutil
import tarfile
import uuid
import requests
import zipfile
from tqdm import tqdm

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
    if config.dataset_url == "":
        return

    path = os.path.join(config.dataset_store_achive_path)

    file_path = os.path.join(
        path, f"{config.dataset_archive_name}.{config.dataset_archive_ext}"
    )

    ensure_file_not_exist(config, file_path)
    os.makedirs(path, exist_ok=True)

    response = requests.get(config.dataset_url, stream=True, timeout=1000)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    print(f"Downloading raw {config.dataset_archive_name} dataset...")
    with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
        with open(file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


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
    print(f"Extracting downloaded {config.dataset_archive_name} archive...")
    path = os.path.join(config.dataset_store_achive_path)
    if os.path.exists(path):
        file_path = os.path.join(
            path, f"{config.dataset_archive_name}.{config.dataset_archive_ext}"
        )
        if os.path.exists(file_path):
            if config.dataset_archive_ext in ["tar", "tar.gz", "zip"]:
                tmp = os.path.join(f"/tmp/{uuid.uuid4()}")
                if config.dataset_archive_ext == "zip":
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(tmp)
                if config.dataset_archive_ext == "tar":
                    with tarfile.open(file_path) as tar:
                        tar.extractall(tmp)
                elif config.dataset_archive_ext == "tar.gz":
                    with tarfile.open(file_path, "r:gz") as tar:
                        tar.extractall(tmp)
                ensure_directory_not_exist(config, config.dataset_directory_train)
                shutil.copytree(
                    os.path.join(tmp, config.dataset_achive_train_path),
                    os.path.join(config.dataset_directory_train),
                )
                ensure_directory_not_exist(config, config.dataset_directory_test)
                shutil.copytree(
                    os.path.join(tmp, config.dataset_achive_test_path),
                    os.path.join(config.dataset_directory_test),
                )

                shutil.rmtree(tmp)
                print("Finished extracting!")
                return
            else:
                raise Exception(f"{config.dataset_archive_ext} format is not supported")
    raise Exception(
        f"Downloaded archive cannot be extracted. Call download_archive function first"
    )
