"""
This scripts provides an API to prepare all the possible dataframes for the different training scripts.
In case of the scripts using AVA dataset, the dataframes are constructed from the AVA.txt file.
As for the dataset of horses, the images are first clustered based on their time data, and then pairs are generated
from within these clusters, always pairing the few approved images with all the different rejected images.
All dataframes are saved as .csv files, and can be found in the _metadata folders inside either
iaa/data/ava or iaa/data/horses.

These dataframes were prepared on a Windows computer, but due to different filename formatting on Linux and Mac,
all of these dataframes should be recreated on these operating systems.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
from iaa.src.utils.clustering import ClusteringEngine


AVA_TEXT_PATH = "../../data/ava/AVA.txt"
AVA_DATASET_TRAIN_PATH = "../../data/ava/dataset/train"
AVA_DATASET_TEST_PATH = "../../data/ava/dataset/test"

HORSES_DATASET_APPROVED_PATH = "../../data/horses/dataset/approved/"
HORSES_DATASET_REJECTED_PATH = "../../data/horses/dataset/rejected/"

SELECTED_CATEGORIES_FOR_GCIAA_CAT_TRAINING = (19, 20, 43, 57, 21, 50, 2, 4, 38, 14, 15, 47, 7, 42, 26)
SEED = 31


DATAFRAME_AVA_GIIAA_HIST_TRAIN_PATH = "../../data/ava/giiaa_metadata/dataframe_AVA_giiaa-hist_train.csv"
DATAFRAME_AVA_GIIAA_HIST_TEST_PATH = "../../data/ava/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"

DATAFRAME_AVA_GCIAA_CAT_TRAIN_PATH = "../../data/ava/gciaa_metadata/dataframe_AVA_gciaa-cat_train.csv"
DATAFRAME_AVA_GCIAA_CAT_TEST_PATH = "../../data/ava/gciaa_metadata/dataframe_AVA_gciaa-cat_test.csv"

DATAFRAME_AVA_GCIAA_DIST_TRAIN_PATH = "../../data/ava/gciaa_metadata/dataframe_AVA_gciaa-dist_train.csv"
DATAFRAME_AVA_GCIAA_DIST_TEST_PATH = "../../data/ava/gciaa_metadata/dataframe_AVA_gciaa-dist_test.csv"

DATAFRAME_HORSES_CLUSTERS_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_clusters.csv"
DATAFRAME_HORSES_PAIRS_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pairs.csv"
DATAFRAME_HORSES_PCIAA_TRAIN_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pciaa_train.csv"
DATAFRAME_HORSES_PCIAA_TEST_PATH = "../../data/horses/pciaa_metadata/dataframe_horses_pciaa_test.csv"


def prepare_dataframe_giiaa_mean(image_dataset_path, image_info_path):
    original_dataframe = pd.read_csv(image_info_path, sep=' ')

    data = {
        "id": [],
        "label": []
    }

    count = 0

    # Loop through image directory and get the row from AVA.txt dataframe where the index
    # (in dataframe) equals filename minus .jpg.
    for filename in os.listdir(image_dataset_path):



        file_index = filename.split('.')[0]

        if file_index.isdigit():
            image_data = original_dataframe[original_dataframe["index"] == int(file_index)].iloc[0]
        else:
            print("Non-digit file name: {}".format(file_index))
            continue

        count += 1

        # Get the average of ranks for now (although literature suggests to use different metrics - e.g. histograms).
        num_annotations = 0
        rank_sum = 0

        for i in range(1, 11):
            rank_sum += image_data[str(i)] * i
            num_annotations += image_data[str(i)]

        average_rank = rank_sum / num_annotations

        data["id"].append(filename)
        data["label"].append(average_rank / 10)

    return pd.DataFrame(data)


def prepare_dataframe_giiaa_hist(image_dataset_path, image_info_path):

    original_dataframe = pd.read_csv(image_info_path, sep=' ')

    data = {
        'id': [],
        'label': []
    }

    # Loop through image directory and get the row from AVA.txt dataframe where the index
    # (in dataframe) equals filename minus .jpg.
    for filename in tqdm(os.listdir(image_dataset_path)):

        file_index = filename.split('.')[0]

        if file_index.isdigit():
            image_data = original_dataframe[original_dataframe['index'] == int(file_index)].iloc[0]
        else:
            print("Non-digit file name: {}".format(file_index))
            continue

        # Get the histogram distribution of annotated scores as a list.
        score_distribution = []
        num_annotations = 0.0

        for i in range(1, 11):
            num_annotations += image_data[str(i)]

        for i in range(1, 11):
            score_distribution.append(image_data[str(i)] / num_annotations)

        data['id'].append(os.path.join(image_dataset_path, filename))
        data['label'].append(score_distribution)

    return pd.DataFrame(data)


def prepare_dataframe_gciaa_cat(
        image_dataset_path,
        image_info_path,
        selected_categories=SELECTED_CATEGORIES_FOR_GCIAA_CAT_TRAINING,
        pairs_per_category_scalar=0.7):

    np.random.seed(SEED)

    original_dataframe = pd.read_csv(image_info_path, sep=' ')

    # Get only the file indices from either the train or the test folder.
    relevant_file_indices = []
    for filename in os.listdir(image_dataset_path):
        image_index = filename.split('.')[0]
        if image_index.isdigit():
            relevant_file_indices.append(image_index)

    # Get only the file indices containing at least one category tag.
    filtered_dataframe = original_dataframe.loc[original_dataframe['index'].isin(relevant_file_indices)
                                                & ((original_dataframe['tag1'] > 0) | (original_dataframe['tag2'] > 0))]

    data = {
        'id_a': [],
        'id_b': [],
        'label': []
    }

    for i in tqdm(selected_categories):

        images_per_category = filtered_dataframe.loc[(filtered_dataframe['tag1'] == i) | (filtered_dataframe['tag2'] == i)]

        print("Number of images for category {}: {}".format(i, len(images_per_category)))

        for ii in range(int(len(images_per_category) * pairs_per_category_scalar)):

            try:
                random_pair = images_per_category.sample(2)
            except ValueError:
                print("Category {} has too few images.".format(i))
                break

            average_ranks = []

            for iii in range(2):
                num_annotations = 0
                rank_sum = 0
                image_in_pair = random_pair.iloc[iii]

                for iv in range(1, 11):
                    rank_sum += image_in_pair[str(iv)] * iv
                    num_annotations += image_in_pair[str(iv)]

                average_ranks.append(rank_sum / num_annotations)

            data['id_a'].append(os.path.join(image_dataset_path, "{}.jpg".format(random_pair.iloc[0]['index'])))
            data['id_b'].append(os.path.join(image_dataset_path, "{}.jpg".format(random_pair.iloc[1]['index'])))
            data['label'].append(float(average_ranks[0] < average_ranks[1]))

    return pd.DataFrame(data)


def prepare_dataframe_gciaa_dist(
        image_dataset_path,
        pairs_per_dataset_scalar=0.25):

    np.random.seed(SEED)

    data = {
        'id': [],
        'label': []
    }

    filenames = os.listdir(image_dataset_path)
    random.shuffle(filenames)

    for filename in tqdm(filenames[:int(len(filenames) * pairs_per_dataset_scalar)]):

        data['id'].append(os.path.join(image_dataset_path, filename))
        data['label'].append(0.0)

    return pd.DataFrame(data)


def prepare_dataframe_pciaa_clusters(image_dataset_approved_path, image_dataset_rejected_path):

    data = {
        'id': [],
        'timestamp': [],
        'approved': []
    }

    for filename in tqdm(os.listdir(image_dataset_approved_path)):
        timestamp = Image.open(os.path.join(image_dataset_approved_path, filename)).getexif()[36867]

        data['id'].append(filename)
        data['timestamp'].append(timestamp)
        data['approved'].append(1.0)

    for filename in tqdm(os.listdir(image_dataset_rejected_path)):
        timestamp = Image.open(os.path.join(image_dataset_rejected_path, filename)).getexif()[36867]

        data['id'].append(filename)
        data['timestamp'].append(timestamp)
        data['approved'].append(0.0)

    engine = ClusteringEngine(data['timestamp'])
    cluster_vector = engine.cluster_chronologically_sorted()

    clustered_data = {
        'id': data['id'],
        'cluster': cluster_vector,
        'approved': data['approved']
    }

    return pd.DataFrame(clustered_data)


def prepare_dataframe_pciaa_pairs(cluster_dataframe_path):

    np.random.seed(SEED)

    cluster_dataframe = pd.read_csv(cluster_dataframe_path)

    generated_pairs = {
        'id_a': [],
        'id_b': [],
        'label': []
    }

    # Generate pairs of images from within the clusters.
    num_clusters = cluster_dataframe["cluster"].max()
    for i in tqdm(range(0, int(num_clusters))):
        current_cluster_dataframe = cluster_dataframe.loc[cluster_dataframe['cluster'] == i]

        approved_filenames = current_cluster_dataframe.loc[current_cluster_dataframe['approved'] == 1.0]['id'].values
        rejected_filenames = current_cluster_dataframe.loc[current_cluster_dataframe['approved'] == 0.0]['id'].values

        if len(approved_filenames) != 0 and len(rejected_filenames) != 0:

            for approved_filename in approved_filenames:
                for rejected_filename in rejected_filenames:

                    if random.random() < 0.5:
                        generated_pairs['id_a'].append(os.path.join(HORSES_DATASET_APPROVED_PATH, approved_filename))
                        generated_pairs['id_b'].append(os.path.join(HORSES_DATASET_REJECTED_PATH, rejected_filename))
                        generated_pairs['label'].append(0.0)
                    else:
                        generated_pairs['id_a'].append(os.path.join(HORSES_DATASET_REJECTED_PATH, rejected_filename))
                        generated_pairs['id_b'].append(os.path.join(HORSES_DATASET_APPROVED_PATH, approved_filename))
                        generated_pairs['label'].append(1.0)

    return pd.DataFrame(generated_pairs)


def split_dataframe_test_train(pairs_dataframe_path, test_split=0.4):
    pairs_dataframe = pd.read_csv(pairs_dataframe_path, index_col=0)

    np.random.seed(SEED)
    mask = np.random.rand(len(pairs_dataframe)) < test_split

    train_split = pairs_dataframe[~mask]
    test_split = pairs_dataframe[mask]

    return train_split, test_split


if __name__ == "__main__":
    dataframe_train, dataframe_test = split_dataframe_test_train(DATAFRAME_HORSES_PAIRS_PATH)
    dataframe_train.to_csv(DATAFRAME_HORSES_PCIAA_TRAIN_PATH)
    dataframe_test.to_csv(DATAFRAME_HORSES_PCIAA_TEST_PATH)
