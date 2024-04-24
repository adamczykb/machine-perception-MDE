import os
import zipfile
from tqdm import tqdm
import subprocess
from cli.config import ProjectConfig
from mde.utils.dialogs import ensure_file_not_exist


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
    if ensure_file_not_exist(config, path):
        subprocess.run(
            f"wget -i {config.dataset_config} -P {path}",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )


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
                            f"{config.dataset_directory}/{config.dataset_archive_name}"
                        )
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
        subprocess.run(
            "find "
            + f"{config.dataset_directory}/{config.dataset_archive_name}/"
            + " -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
