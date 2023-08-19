import unittest
from unittest.mock import patch, MagicMock
from ASL_alphabet.components import DataIngestion
from ASL_alphabet import logger
from ASL_alphabet.entity import DataIngestionConfig
from zipfile import ZipFile  # Import ZipFile directly from zipfile module
import urllib.request as request

class TestDataIngestionIntegration(unittest.TestCase):

    @patch('os.path.exists', side_effect=[False, True])  # Corrected side_effect
    @patch('urllib.request.urlretrieve')
    @patch('zipfile.ZipFile')
    @patch('ASL_alphabet.logger.info')
    def test_download_and_extract(self, mock_logger_info, mock_zipfile, mock_urlretrieve, mock_os_path_exists):
        # Mock configuration
        config = DataIngestionConfig(
            root_dir="path/to/data",
            source_URL="http://example.com/data.zip",
            local_data_file="path/to/local_data.zip",
            unzip_dir="path/to/unzipped_data"
        )

        # Create a DataIngestion instance
        data_ingestion = DataIngestion(config)

        # Mock file download
        mock_urlretrieve.return_value = ("path/to/local_data.zip", "headers")
        data_ingestion.download_file()

        # Assert that download was attempted
        mock_logger_info.assert_any_call("Trying to download file...")
        mock_urlretrieve.assert_called_once_with(url="http://example.com/data.zip", filename="path/to/local_data.zip")
        mock_logger_info.assert_any_call("path/to/local_data.zip download! with following info: \nheaders")

        # Mock unzip operation
        mock_zipfile_instance = MagicMock()
        mock_zipfile.return_value = mock_zipfile_instance
        data_ingestion.check_and_extract_videos()

        # Assert that unzip was attempted only when the unzip directory exists
        mock_logger_info.assert_any_call("extraction completed")
        mock_zipfile.assert_called_once_with("path/to/local_data.zip", 'r')
        mock_zipfile_instance.extractall.assert_called_once_with("path/to/unzipped_data")

    @patch('ASL_alphabet.components.os.path.exists', return_value=False)  # Corrected patch setup
    @patch('ASL_alphabet.logger.info')
    def test_check_and_extract_empty_folder(self, mock_logger_info, mock_os_path_exists):
        # Mock configuration
        config = DataIngestionConfig(
            root_dir="path/to/data",
            source_URL="http://example.com/data.zip",
            local_data_file="path/to/local_data.zip",
            unzip_dir="path/to/unzipped_data"
        )

        # Create a DataIngestion instance
        data_ingestion = DataIngestion(config)

        # Mock unzip operation
        data_ingestion.check_and_extract_videos()

        # Assert that extraction was not attempted for a non-empty folder
        mock_logger_info.assert_any_call("error in Extraction...")
        mock_os_path_exists.assert_called_once_with("path/to/unzipped_data")

if __name__ == "__main__":
    unittest.main()
