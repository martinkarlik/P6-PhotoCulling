"""
This script solves the problem of clustering data based on its time information.

Image's time metadata is expected to follow a format "YY/MM/DD HH:MM:SS",
although this script is adaptable and adding or removing information (say changing the format,
removing year data, or adding milliseconds or the image's file index) is a minor implementation change.

Input: An array (any size) of textual time data information in the format above.
Output: A key-value store encoding the cluster information. Keys are cluster indices (positive integers),
values are lists containing chronologically ordered time data belonging to a cluster.

Algorithm:
    * We use distance between data stamps to determine the neighbours, which should be clustered together.
    * The distance is represented as a tuple: (days, seconds), see DISTANCE THRESHOLD below. It is something
      like a priority-ordered distance tuple. If extra information is necessary (file index when time data missing,
      or some other metadata), it can simply be added to the distance representation, e.g. (days, seconds, file_index)
      The representation of distance using a uniform scale, say seconds or milliseconds,
      could cause data overflow when dealing with a huge distance, that's why a tuple. (One year is 3.2 × 10^10 milliseconds,
      we cannot store that in a numeric variable. There probably won't be a one year difference between our
      images, but when solving a problem, it's a good practice to generalize it to its extreme.)
    * We loop through all time stamps and compare them with the ones already clustered. If a time stamp is
      close enough (less than DISTANCE_THRESHOLD) to any other time stamp in a cluster,
      the time stamp is put into that cluster. If it belongs to no cluster, it is put to a new cluster.

Possible improvements:
    * Cluster content is ordered chronologically, but not the clusters themselves.
    * Other metadata, such as image's file index could be considered, when image's time data is ambigous or missing.
    * Distance threshold could adapt to the context.
    * This is quite a basic implementation. (I googled time series clustering, but what I found seemed to be
      unnecessarily complicated for our problem, so I wrote this for now.)
"""

import datetime as dt
import numpy as np
from tqdm import tqdm
import functools

FORMAT = "%Y:%m:%d %H:%M:%S"
DISTANCE_THRESHOLD = (0, 60)  # days, seconds


class ClusteringEngine:

    def __init__(self, data):
        self.data = data

    def _get_distance(self, timestamp_a, timestamp_b):

        distance = dt.datetime.strptime(timestamp_b, FORMAT) - \
                   dt.datetime.strptime(timestamp_a, FORMAT)
        return [distance.days, distance.seconds]

    def _is_predecessor(self, timestamp_a, timestamp_b):
        """Did timestamp_a come before timestamp_b?"""

        distance = self._get_distance(timestamp_a, timestamp_b)

        for i in range(0, len(distance)):
            if distance[i] != 0:
                return distance[i] > 0

        # If two time stamps are equal, (i) this should never happen (ii) let's just return True
        return True

    def _in_proximity(self, timestamp_a, timestamp_b, distance_threshold=DISTANCE_THRESHOLD):
        """Is one time stamp close to another time stamp with respect to distance threshold?"""

        distance = self._get_distance(timestamp_a, timestamp_b) if \
            self._is_predecessor(timestamp_a, timestamp_b) else \
            self._get_distance(timestamp_b, timestamp_a)

        result = True

        for i in range(0, len(distance)):
            if abs(distance[i]) > distance_threshold[i]:
                result = False
                break

        return result

    def _insert_chronologically(self, example, cluster):
        """This function inserts a datapoint into a cluster chronologically."""

        is_timestamp_inserted = False
        for i in range(0, len(cluster)):
            if self._is_predecessor(example, cluster[i]):
                cluster.insert(i, example)
                is_timestamp_inserted = True

        if not is_timestamp_inserted:
            cluster.append(example)

    def cluster_chronologically(self):
        """The actual algorithm."""

        clusters = {}

        for timestamp in tqdm(self.data):

            is_timestamp_sorted = False

            i = 0
            while i < len(clusters) and not is_timestamp_sorted:

                j = 0
                while j < len(clusters[i]) and not is_timestamp_sorted:
                    if self._in_proximity(timestamp, clusters[i][j]):
                        self._insert_chronologically(timestamp, clusters[i])
                        is_timestamp_sorted = True
                    j += 1

                i += 1

            if not is_timestamp_sorted:
                clusters[len(clusters)] = [timestamp]

        cluster_vector = np.zeros(len(self.data))

        for key in tqdm(clusters):

            for timestamp in clusters[key]:
                cluster_vector[np.where(self.data == timestamp)] = key

        return cluster_vector.tolist()

    def cluster_chronologically_sorted(self):

        def comparator(a, b):
            result = -1 if self._is_predecessor(a, b) else 1
            return result

        sorted_data = sorted(self.data, key=functools.cmp_to_key(comparator))
        cluster_vector = np.zeros(len(self.data))

        cluster_index = 0
        for i in tqdm(range(len(sorted_data) - 1)):

            indices = [ii for ii in range(len(self.data)) if self.data[ii] == sorted_data[i]]
            for ii in indices:
                cluster_vector[ii] = cluster_index

            if not self._in_proximity(sorted_data[i], sorted_data[i + 1]):
                cluster_index += 1

        print(cluster_vector)

        return cluster_vector.tolist()


    def get_cluster_vector(self):
        if self.clusters is None:
            return



