import unittest
import os
from ASL_alphabet.config import ConfigurationManager
from ASL_alphabet.entity import DataIngestionConfig, PrepareBaseModelConfig
import yaml
from ASL_alphabet.constants import *
from pathlib import Path


class TestConfigurationManagerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Creating temporary directories and files for testing
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
            "prepare_base_model": {
                "root_dir": "model",
                "base_model_path": "path/to/base_model.h5",
                "updated_base_model_path": "path/to/updated_base_model.h5"
            },
            "artifacts_root": self.temp_artifacts_dir
        }

        self.temp_params_data = {
            "KERNEL_LAYERS1": 64,
            "KERNEL_LAYERS2": 128,
            "KERNEL_LAYERS3": 256,
            "CLASSES": 26,
            "FINAL_OUTPUT_LAYERS": 26,
            "AUGMENTATION": True,
            "IMAGE_SIZE": [64, 64],  # Use list instead of tuple
            "BATCH_SIZE": 32,
            "EPOCHS": 10
        }

        # Writing temporary data to files
        with open(self.temp_config_file, "w") as config_file:
            yaml.dump(self.temp_config_data, config_file)

        with open(self.temp_params_file, "w") as params_file:
            yaml.dump(self.temp_params_data, params_file)

    # ... Rest of the test class remains unchanged ...


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
        prepare_base_model_config = config_manager.get_prepare_base_model_config()

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

        # Verifying that the PrepareBaseModelConfig object is created correctly
        self.assertIsInstance(prepare_base_model_config, PrepareBaseModelConfig)
        self.assertEqual(prepare_base_model_config.root_dir, Path("model"))
        self.assertEqual(prepare_base_model_config.base_model_path, Path("path/to/base_model.h5"))
        self.assertEqual(prepare_base_model_config.updated_base_model_path, Path("path/to/updated_base_model.h5"))
        self.assertEqual(prepare_base_model_config.kernel_layers1, self.temp_params_data["KERNEL_LAYERS1"])
        self.assertEqual(prepare_base_model_config.kernel_layers2, self.temp_params_data["KERNEL_LAYERS2"])
        self.assertEqual(prepare_base_model_config.kernel_layers3, self.temp_params_data["KERNEL_LAYERS3"])
        self.assertEqual(prepare_base_model_config.params_classes, self.temp_params_data["CLASSES"])
        self.assertEqual(prepare_base_model_config.params_final_output_layers, self.temp_params_data["FINAL_OUTPUT_LAYERS"])
        self.assertEqual(prepare_base_model_config.params_augmentation, self.temp_params_data["AUGMENTATION"])
        self.assertEqual(prepare_base_model_config.params_image_size, self.temp_params_data["IMAGE_SIZE"])
        self.assertEqual(prepare_base_model_config.params_batch_size, self.temp_params_data["BATCH_SIZE"])
        self.assertEqual(prepare_base_model_config.params_epochs, self.temp_params_data["EPOCHS"])


if __name__ == "__main__":
    unittest.main()
