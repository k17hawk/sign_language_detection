import unittest
import os
from ASL_alphabet.config import ConfigurationManager
from ASL_alphabet.entity import DataIngestionConfig
import yaml
from ASL_alphabet.constants import *
from pathlib import Path
class TestConfigurationManagerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Creating  temporary directories and files for testing
        self.temp_config_file = Path("tmp/temp_config.yaml")
        self.temp_params_file = Path("tmp/temp_params.yaml")
        self.temp_artifacts_dir = "temp_artifacts"
        os.makedirs(self.temp_artifacts_dir, exist_ok=True)
        

        self.temp_config_data = {
            "data_ingestion": {
                "root_dir": "data",
                "source_URL": "https://www.kaggle.com/datasets/kapillondhe/american-sign-language/download?datasetVersionNumber=1",                  
                "local_data_file": "archive.zip",
                "unzip_dir": "unzipped_data"
            },
            "artifacts_root": self.temp_artifacts_dir
        }

        self.temp_params_data = {
            # Your temporary parameter data, if needed for testing
        }

        # Writing temporary data to files
        with open(self.temp_config_file, "w") as config_file:
            yaml.dump(self.temp_config_data, config_file)

        with open(self.temp_params_file, "w") as params_file:
            yaml.dump(self.temp_params_data, params_file)

    @classmethod
    def tearDownClass(self):
        # Removing temporary directories and files after testing
        os.remove(self.temp_config_file)
        os.remove(self.temp_params_file)
        os.rmdir(self.temp_artifacts_dir)

    def test_configuration_manager(self):
        # Arrange
        config_filepath = self.temp_config_file
        params_filepath = self.temp_params_file

        # Act
        config_manager = ConfigurationManager(config_filepath, params_filepath)
        data_ingestion_config = config_manager.get_data_ingestion_config()

        # Assert
        # Verifying that the config and params were loaded properly
        self.assertEqual(config_manager.config, self.temp_config_data)
        self.assertEqual(config_manager.params, self.temp_params_data)

        # Verifying that the directories were created as expected
        self.assertTrue(os.path.isdir(self.temp_artifacts_dir))
        self.assertTrue(os.path.isdir(self.temp_config_data["data_ingestion"]["root_dir"]))

        # Verifying that the DataIngestionConfig object is created correctly
        self.assertIsInstance(data_ingestion_config, DataIngestionConfig)
        self.assertEqual(data_ingestion_config.root_dir, self.temp_config_data["data_ingestion"]["root_dir"])
        self.assertEqual(data_ingestion_config.source_URL, self.temp_config_data["data_ingestion"]["source_URL"])
        self.assertEqual(data_ingestion_config.local_data_file, self.temp_config_data["data_ingestion"]["local_data_file"])
        self.assertEqual(data_ingestion_config.unzip_dir, self.temp_config_data["data_ingestion"]["unzip_dir"])

if __name__ == "__main__":
    unittest.main()
