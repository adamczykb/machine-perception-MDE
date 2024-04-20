from cli.config import ProjectConfig

import yaml


class ConfigParser:
    @staticmethod
    def read( path) -> ProjectConfig:
        with open(path,'rb') as stream:
            pc = ProjectConfig()
            yml_config = yaml.safe_load(stream)
            pc.dataset_url = yml_config["data"]["url"]
            pc.dataset_directory_train = yml_config["data"]["dataset"][
                "directory_train"
            ]
            pc.dataset_directory_test = yml_config["data"]["dataset"]["directory_test"]
            pc.dataset_archive_ext = yml_config["data"]["dataset"]["archive"]['ext']
            pc.dataset_archive_name = yml_config["data"]["dataset"]["archive"]['name']
            pc.dataset_store_achive_path = yml_config["data"]["dataset"]["archive"]['store_path']
            pc.dataset_achive_train_path = yml_config["data"]["dataset"]["archive"]['train_path']
            pc.dataset_achive_test_path = yml_config["data"]["dataset"]["archive"]['test_path']
            pc.verbose = yml_config["verbose"]
            pc.model_name=yml_config['model_name']
            return pc
