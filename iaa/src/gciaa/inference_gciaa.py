"""
Using the differently trained GCIAA models to make inference on a few random samples from the AVA test folder.
"""

import pandas as pd
import cv2
import os
import random
from iaa.src.gciaa.base_module_gciaa import *
from iaa.src.giiaa.base_module_giiaa import *
import tensorflow.keras as keras


GIIAA_MODEL = "../../models/giiaa-hist_204k_base-inceptionresnetv2_loss-0.078.hdf5"
GCIAA_CATEGORIES_MODEL = "../../models/gciaa-cat_81k_base-giiaa_accuracy-0.710.hdf5"
GCIAA_DISTORTIONS_MODEL = "../../models/gciaa-dist_51k_base-giiaa_accuracy-0.906.hdf5"
PCIAA_HORSES_MODEL = "../../models/pciaa_horses_90k-pairs_accuracy-0.951.hdf5"

AVA_DATASET_TEST_PATH = "../../data/ava/dataset/test"
AVA_DATAFRAME_TEST_PATH = "../../data/ava/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"


def get_mean(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean


if __name__ == "__main__":

    giiaa = keras.models.load_model(GIIAA_MODEL, custom_objects={"earth_movers_distance": earth_movers_distance})

    gciaa = BaseModule(weights=GCIAA_CATEGORIES_MODEL, load_weights_as='GCIAA')
    gciaa.build()
    gciaa.compile()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    for i in range(20):

        random_file = os.path.join(AVA_DATASET_TEST_PATH, random.choice(os.listdir(AVA_DATASET_TEST_PATH)))
        image_a = cv2.resize(cv2.imread(random_file), (224, 224)) / 255.0
        image_a = np.asarray(image_a)[np.newaxis, ...]
        gt_a = dataframe[dataframe['id'] == random_file.split('/')[-1]].iloc[0]['label']

        random_file = os.path.join(AVA_DATASET_TEST_PATH, random.choice(os.listdir(AVA_DATASET_TEST_PATH)))
        image_b = cv2.resize(cv2.imread(random_file), (224, 224)) / 255 * 0.01
        image_b = np.asarray(image_b)[np.newaxis, ...]
        gt_b = dataframe[dataframe['id'] == random_file.split('/')[-1]].iloc[0]['label']

        giiaa_prediction_a = get_mean(giiaa.predict(image_a)[0])
        giiaa_prediction_b = get_mean(giiaa.predict(image_b)[0])

        giiaa_gt_a = get_mean(gt_a)
        giiaa_gt_b = get_mean(gt_b)

        gciaa_prediction = gciaa.predict([image_a, image_b])[0, 0]

        print("GT A {:.2f} | GT B {:.2f} | GIIAA A {:.2f} | GIIAA B {:.2f} | GCIAA {:.2f}"
              .format(giiaa_gt_a, giiaa_gt_b, giiaa_prediction_a, giiaa_prediction_b, gciaa_prediction))

