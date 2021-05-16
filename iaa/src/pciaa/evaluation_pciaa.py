"""
Evaluation of PCIAA within-cluster classification.
Evaluated on 61 894 out of 155 532 within-cluster pairs of images of horses,
generated from a private dataset of a professional photographer.

                Baseline    |   PCIAA
Loss:           0.0000      |   0.0000
Accuracy:       0.0000      |   0.0000
"""


from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

GIIAA_MODEL = "../../models/giiaa_metadata/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"
PCIAA_HORSES_MODEL = ""

HORSES_DATAFRAME_TEST_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pciaa_test.csv"

BASE_MODEL_NAME = "InceptionResNetV2"
BATCH_SIZE = 64


if __name__ == "__main__":

    # model = keras.models.load_model(WEIGHTS_PATH, custom_objects={"earth_movers_distance": earth_movers_distance})
    base = BaseModule(
        base_model_name=BASE_MODEL_NAME,
        weights=PCIAA_HORSES_MODEL)

    base.build()
    base.compile()

    dataframe = pd.read_csv(HORSES_DATAFRAME_TEST_PATH, converters={'label': eval})

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

