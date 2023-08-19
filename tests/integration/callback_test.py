import unittest
import os
from ASL_alphabet.config import ConfigurationManager
from ASL_alphabet.entity import PrepareCallbacksConfig
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
            "prepare_callbacks": {
                "root_dir": "path/to/root_dir",
                "tensorboard_root_log_dir": "path/to/tensorboard_dir",
                "checkpoint_model_filepath": "path/to/model_checkpoint.ckpt"
            },
            "artifacts_root": self.temp_artifacts_dir
        }

        # Writing temporary data to files
        with open(self.temp_config_file, "w") as config_file:
            yaml.dump(self.temp_config_data, config_file)

    # ... Rest of the test class remains unchanged ...

    @classmethod
    def tearDownClass(self):
        # Removing temporary directories and files after testing
        os.remove(self.temp_config_file)
        os.rmdir(self.temp_artifacts_dir)

    def test_get_prepare_callback_config(self):
        # Arrange
        config_filepath = self.temp_config_file
        config_manager = ConfigurationManager(config_filepath, self.temp_params_file)

        # Act
        prepare_callback_config = config_manager.get_prepare_callback_config()

        # Assert
        # Verifying that the config was loaded properly
        self.assertEqual(config_manager.config, self.temp_config_data)

        # Verifying that the directories were created as expected
        self.assertTrue(os.path.isdir(self.temp_artifacts_dir))

        # Verifying that the PrepareCallbacksConfig object is created correctly
        self.assertIsInstance(prepare_callback_config, PrepareCallbacksConfig)
        self.assertEqual(prepare_callback_config.root_dir, Path("path/to/root_dir"))
        self.assertEqual(prepare_callback_config.tensorboard_root_log_dir, Path("path/to/tensorboard_dir"))
        self.assertEqual(prepare_callback_config.checkpoint_model_filepath, Path("path/to/model_checkpoint.ckpt"))


if __name__ == "__main__":
    unittest.main()
