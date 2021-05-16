"""
Evaluation of within-category trained GCIAA model.
Evaluated on 20 231 within-category generated pairs from AVA dataset.
Performance of the within-category trained GCIAA model is compared
with the baseline GCIAA model with the GIIAA model as the image encoder.

                Baseline    |   Within-category trained GCIAA
Loss:           0.4274      |   0.0000
Accuracy:       0.6751      |   0.0000
"""

from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

GIIAA_MODEL = "../../models/giiaa_metadata/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"
GCIAA_CATEGORIES_MODEL = ""  # Should be the saved GCIAA weights

AVA_DATASET_TEST_PATH = "../../data/ava/dataset/test/"
AVA_DATAFRAME_TEST_PATH = "../../data/ava/gciaa_metadata/AVA_gciaa-cat_test_dataframe.csv"

BASE_MODEL_NAME = "InceptionResNetV2"
BATCH_SIZE = 64


if __name__ == "__main__":

    # model = keras.models.load_model(WEIGHTS_PATH, custom_objects={"earth_movers_distance": earth_movers_distance})
    base = BaseModule(weights=GCIAA_CATEGORIES_MODEL)
    base.build()
    base.compile()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = SiameseGeneratorCategories(
        generator=data_generator,
        dataframe=dataframe
    )

    accuracy = base.siamese_model.evaluate_generator(
        generator=test_generator.get_pairwise_flow_from_dataframe(),
        steps=test_generator.samples_per_epoch / test_generator.batch_size,
        verbose=1
    )

