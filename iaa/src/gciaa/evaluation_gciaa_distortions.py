"""
Evaluation of distortion-trained GCIAA model.
Evaluated on 12 775 distortion-augmented pairs generated from AVA dataset.
Distortions are evaluated individually, the performance of the distortion-trained GCIAA model
is compared with the baseline GCIAA model with GIIAA model as the image encoder.

UNDEREXPOSURE   Baseline    |   Distortion trained GCIAA
Loss:           0.4432      |   0.0000
Accuracy:       0.7948      |   0.0000

OVEREXPOSURE    Baseline    |   Distortion trained GCIAA
Loss:           0.4512      |   0.0000
Accuracy:       0.7531      |   0.0000

ROTATION        Baseline    |   Distortion trained GCIAA
Loss:           0.3825      |   0.0000
Accuracy:       0.9742      |   0.0000

SHEARING        Baseline    |   Distortion trained GCIAA
Loss:           0.0000      |   0.0000
Accuracy:       0.0000      |   0.0000

ZOOMING         Baseline    |   Distortion trained GCIAA
Loss:           0.0000      |   0.0000
Accuracy:       0.0000      |   0.0000

BLUR            Baseline    |   Distortion trained GCIAA
Loss:           0.0000      |   0.0000
Accuracy:       0.0000      |   0.0000

BLOBS           Baseline    |   Distortion trained GCIAA
Loss:           0.0000      |   0.0000
Accuracy:       0.0000      |   0.0000

----------------------------------------------------------

ALL DISTORTIONS Baseline    |   Distortion trained GCIAA
Loss:           avg         |   avg
Accuracy:       avg         |   avg
"""


from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.utils.generators import *
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


GIIAA_MODEL = "../../models/giiaa/model_giiaa-hist_200k_inceptionresnetv2_0.078.hdf5"
GCIAA_DISTORTIONS_MODEL = ""

AVA_DATASET_TEST_PATH = "../../datasets/ava/test/"
AVA_DATAFRAME_TEST_PATH = "../../datasets/ava/gciaa/AVA_gciaa-dist_test_dataframe.csv"

BASE_MODEL_NAME = "InceptionResNetV2"
BATCH_SIZE = 64

DISTORTIONS = [
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[0.2, 0.75]),  # Underexposure
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[1.5, 5.0]),   # Overexposure
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, rotation_range=90.0),           # Rotation
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, shear_range=90.0),              # Shearing
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, zoom_range=0.5),                # Zooming
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2,
                       preprocessing_function=SiameseGeneratorDistortions.apply_blur),          # Blur
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2,
                       preprocessing_function=SiameseGeneratorDistortions.apply_blob)           # Blob
]


if __name__ == "__main__":

    # model = keras.models.load_model(WEIGHTS_PATH, custom_objects={"earth_movers_distance": earth_movers_distance})
    base = BaseModule(
        base_model_name=BASE_MODEL_NAME,
        weights=GIIAA_MODEL)

    base.build()
    base.compile()
    base.siamese_model.summary()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    distortion_generators = [
        DISTORTIONS[3]
    ]

    test_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe)

    accuracy = base.siamese_model.evaluate_generator(
        generator=test_generator.get_pairwise_flow_from_dataframe(),
        steps=test_generator.samples_per_epoch / test_generator.batch_size,
        verbose=1
    )

