"""
Evaluation of the GIIAA model.
"""


from iaa.src.giiaa.base_module_giiaa import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = "../../models/giiaa-hist_204k_base-inceptionresnetv2_loss-0.078.hdf5"

AVA_DATASET_TEST_PATH = "../../data/ava/dataset/test/"
AVA_DATAFRAME_TEST_PATH = "../../data/ava/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"

BATCH_SIZE = 32


def get_mean(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)

    return mean


if __name__ == "__main__":

    nima = NimaModule()
    nima.build()
    nima.nima_model.load_weights(MODEL_PATH)
    nima.nima_model.compile()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = data_generator.flow_from_dataframe(
        directory=AVA_DATASET_TEST_PATH,
        dataframe=dataframe,
        x_col='id',
        y_col=['label'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    nima.nima_model.evaluate_generator(
        generator=test_generator,
        steps=test_generator.samples / test_generator.batch_size,
        verbose=1
    )

