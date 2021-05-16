"""
Training script for distribution-based GCIAA.
"""

from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
import pandas as pd
import os


AVA_DATAFRAME_PATH = "../../data/ava/gciaa_metadata/AVA_gciaa-dist_train_dataframe.csv"

GIIAA_MODEL = "../../models/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"

MODELS_PATH = "../../models"
MODEL_NAME_TAG = 'gciaa-dist_51k_base-giiaa'


LOG_PATH = "../../data/ava/gciaa_metadata/gciaa-dist_logs"
BASE_MODEL_NAME = "InceptionResNetV2"
BATCH_SIZE = 96
DROPOUT_RATE = 0.75
USE_MULTIPROCESSING = False
N_WORKERS = 1

EPOCHS = 5


if __name__ == "__main__":

    tensorboard = TensorBoard(
        log_dir=LOG_PATH, update_freq='batch'
    )

    model_save_name = (MODEL_NAME_TAG + '_{accuracy:.3f}.hdf5')
    model_file_path = os.path.join(MODELS_PATH, model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    base = BaseModule(weights=GIIAA_MODEL)
    base.build()
    base.compile()

    # Training the GCIAA model with artificially created pairs of images.
    dataframe = pd.read_csv(AVA_DATAFRAME_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    distortion_generators = [
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[0.2, 0.75]),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[1.5, 5.0]),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, rotation_range=90.0),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, shear_range=90.0),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, zoom_range=0.5),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, preprocessing_function=SiameseGeneratorDistortions.apply_blur),
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, preprocessing_function=SiameseGeneratorDistortions.apply_blob_overlay)
    ]

    train_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='training')

    validation_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='validation')


    base.siamese_model.fit(
        train_generator.get_pairwise_flow_from_dataframe(),
        # steps_per_epoch=train_generator.samples_per_epoch // train_generator.batch_size,
        steps_per_epoch=5,
        validation_data=validation_generator.get_pairwise_flow_from_dataframe(),
        validation_steps=validation_generator.samples_per_epoch // validation_generator.batch_size,
        epochs=EPOCHS,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=N_WORKERS,
        verbose=1,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer]
    )

    K.clear_session()
