from dataclasses import dataclass


@dataclass
class ProjectConfig:
    dataset_url: str=''
    dataset_directory_train: str=''
    dataset_directory_test: str=''
    dataset_archive_ext: str=''
    dataset_archive_name:str=''
    dataset_store_achive_path:str='/tmp'
    dataset_achive_train_path:str=''
    dataset_achive_test_path:str=''
    verbose:bool=True