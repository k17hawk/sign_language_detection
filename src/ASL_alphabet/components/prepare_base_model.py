import os
import urllib.request as request
from ASL_alphabet.entity import PrepareBaseModelConfig
import tensorflow as tf
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def create_base_model(self):

        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.config.params_image_size),
                tf.keras.layers.Conv2D(
                        self.config.kernel_layers1,(3, 3), 
                        strides=1,padding='same', 
                        activation="relu"
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    padding='same'
                ),
                tf.keras.layers.Conv2D(self.config.kernel_layers2,(3,3),strides=1, padding='same', activation="relu"),
                tf.keras.layers.Dropout(
                    0.2
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    padding='same'
                ),
                tf.keras.layers.Conv2D(self.config.kernel_layers3,(3,3),strides=1, padding='same', activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    padding='same'
                ),
            ]
        )
 
        self.save_model(path=self.config.base_model_path, model=self.model)
        
       



    @staticmethod
    def _prepare_full_model(model, classes,final_output_layers):
   
        flatten_in = tf.keras.layers.Flatten()(model.output)
        flatten_in = tf.keras.layers.Dense(units=classes,activation='relu')(flatten_in)
        flatten_in = tf.keras.layers.Dropout(0.3)(flatten_in)
        prediction = tf.keras.layers.Dense(
            units=final_output_layers,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            final_output_layers=self.config.params_final_output_layers,

        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)