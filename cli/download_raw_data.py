from cli.utils.config_parser import ConfigParser
import sys

from mde.data.download_utils import download_archive, extract_archive

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = ConfigParser().read(sys.argv[1])
    else:
        config = ConfigParser().read("/app/config.yml")
    download_archive(config)
    extract_archive(config)
