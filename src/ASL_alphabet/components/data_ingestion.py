import os
import urllib.request as request
from zipfile import ZipFile
from ASL_alphabet.entity import DataIngestionConfig
from ASL_alphabet import logger
from ASL_alphabet.utils import get_size
from tqdm import tqdm
from pathlib import Path

import zipfile


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        logger.info("Trying to download file...")
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.info("Download started...")
                filename, headers = request.urlretrieve(
                    url="https://www.kaggle.com/datasets/kapillondhe/american-sign-language/download?datasetVersionNumber=1",
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} downloaded! with following info: \n{headers}")

                # Check if the downloaded file is a valid zip file
                if not zipfile.is_zipfile(filename):
                    os.remove(filename)  # Remove the invalid file
                    raise ValueError("Downloaded file is not a zip file.")

            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        except Exception as e:
            raise ValueError(f"Error occurred while downloading the file: {e}")


    def check_and_extract_videos(self):
        """
        Check if the artifact/videos folder is empty. If empty, unzip video data from WLASL_videos.zip
        and move only video files into artifact/videos. Then delete the unzip folder. If not empty, return.
        """
        try:
    
            if os.path.exists(self.config.unzip_dir):
                # Unzip video data from WLASL_videos.zip
                with ZipFile(self.config.local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(self.config.unzip_dir)
                logger.info("extraction completed")
            else:
                logger.info("error in Extraction...")
        except Exception as e:
            raise  e

