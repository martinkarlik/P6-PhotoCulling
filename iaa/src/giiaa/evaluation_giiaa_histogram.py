"""
Evaluation of the GIIAA model.
Evaluated on 51 100 images from AVA dataset.

Earth mover's distance loss: 0.0772
"""


from iaa.src.giiaa.base_module_giiaa import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


GIIAA_PATH = "../../models/giiaa-hist_204k_base-inceptionresnetv2_loss-0.078.hdf5"
AVA_DATASET_TEST_PATH = "../../data/ava/dataset/test"
AVA_DATAFRAME_TEST_PATH = "../../data/ava/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"

BATCH_SIZE = 1


if __name__ == "__main__":

    nima = NimaModule()
    nima.build()
    nima.nima_model.load_weights(GIIAA_PATH)
    nima.compile()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})
    data_generator = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='id',
        y_col=['label'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    nima.nima_model.evaluate_generator(
        generator=test_generator,
        steps=test_generator.samples / test_generator.batch_size,
        verbose=1
    )


