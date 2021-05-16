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


HORSES_DATAFRAME_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pciaa_train.csv"

GCIAA_MODEL = ""
LOG_PATH = "../../data/ava/gciaa_metadata/logs"
MODELS_PATH = "../../models/gciaa/"
MODEL_NAME_TAG = "pciaa_horses_90k-pairs"

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

    base = BaseModule(
        base_model_name=BASE_MODEL_NAME,
        weights=GCIAA_MODEL)
    base.build()
    base.compile()

    # Training the GCIAA model with same-category pairs generated from the AVA dataset.
    dataframe = pd.read_csv(HORSES_DATAFRAME_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = SiameseGeneratorCategories(
        generator=data_generator,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='training')

    validation_generator = SiameseGeneratorCategories(
        generator=data_generator,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='validation')

    base.siamese_model.fit_generator(
        generator=train_generator.get_pairwise_flow_from_dataframe(),
        steps_per_epoch=train_generator.samples_per_epoch // train_generator.batch_size,
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
