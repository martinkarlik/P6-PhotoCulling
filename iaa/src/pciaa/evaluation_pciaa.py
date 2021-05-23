"""
Evaluation of Personalized-Comparative IAA within-cluster classification.
Evaluated on 61 894 out of 155 532 within-cluster pairs of images of horses,
generated from a private dataset of a professional photographer.

                Baseline    |   GCIAA Categories    |   GCIAA Distortions   |   PCIAA
Loss:           0.4720      |   0.0000              |   0.4643              |   0.1070
Accuracy:       0.7290      |   0.0000              |   0.5461              |   0.9086
"""


from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

GIIAA_MODEL = "../../models/giiaa-hist_204k_base-inceptionresnetv2_loss-0.078.hdf5"
GCIAA_CATEGORIES_MODEL = "../../models/gciaa-cat_81k_base-giiaa_accuracy-0.710.hdf5"
GCIAA_DISTORTIONS_MODEL = "../../models/gciaa-dist_51k_base-giiaa_accuracy-0.906.hdf5"
PCIAA_HORSES_MODEL = "../../models/pciaa_horses_90k-pairs_accuracy-0.951.hdf5"

HORSES_DATAFRAME_TEST_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pciaa_test.csv"


if __name__ == "__main__":

    base = BaseModule(weights=GCIAA_CATEGORIES_MODEL, load_weights_as='GCIAA')
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

