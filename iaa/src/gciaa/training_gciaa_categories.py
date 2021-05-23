"""
Training script for within-category-based GCIAA.
This script is based on generating pairs of images from the AVA dataset, which belong to the same AVA categories (or tags).
There are 66 tags in the AVA dataset, and each image is assigned up to two of these tags. We chose 15 out of these categories,
which seemed to contain the most similar images (e.g. tag "Animal" was chosen over "Macro" or "Abstract").
The images' mean aesthetics score was used to determine which of the two images is more aesthetic.
"""

from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
import pandas as pd
import os


AVA_DATAFRAME_PATH = "../../data/ava/gciaa_metadata/dataframe_AVA_gciaa-cat_train.csv"
GIIAA_MODEL = "../../models/giiaa-hist_204k_base-inceptionresnetv2_loss-0.078.hdf5"

LOG_PATH = "../../data/ava/gciaa_metadata/gciaa-cat_logs"
MODELS_PATH = "../../models/"
MODEL_NAME_TAG = "gciaa-cat_81k_base-giiaa"

BATCH_SIZE = 96
DROPOUT_RATE = 0.75
USE_MULTIPROCESSING = False
N_WORKERS = 1

EPOCHS = 5


if __name__ == "__main__":

    tensorboard = TensorBoard(
        log_dir=LOG_PATH, update_freq='batch'
    )

    model_save_name = (MODEL_NAME_TAG + '_accuracy-{accuracy:.3f}.hdf5')
    model_file_path = os.path.join(MODELS_PATH, model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    base = BaseModule(weights=GIIAA_MODEL, load_weights_as='GIIAA')
    base.build()
    base.compile()

    # Training the GCIAA model with same-category pairs generated from the AVA dataset.
    dataframe = pd.read_csv(AVA_DATAFRAME_PATH, converters={'label': eval})

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
