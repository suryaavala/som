import numpy as np


class SOM:

    def __init__(self, Size_X, Size_Y, dim_of_input_vec):
        self.network_dimensions = (Size_X, Size_Y, dim_of_input_vec)
        self.network = np.random.random(self.network_dimensions)
        return

    def _find_bmu(self, current_input_vector):
        # list_euc_distances[euclidean_dist] = network_idx
        # if two are more neurons have the same bmu then the one that was discovered last would be chosen
        list_euc_distances = {}
        for x in range(self.network_dimensions[0]):
            for y in range(self.network_dimensions[1]):
                current_euclidean_distance = (sum(
                    [(current_input_vector[i]-self.network[x][y][i])**2 for i in range(self.network_dimensions[-1])]))**0.5
                list_euc_distances[current_euclidean_distance] = (x, y)
        return list_euc_distances[min(list(list_euc_distances.keys()))]

    def _np_find_bmu(self, current_input_vector):
        return np.unravel_index(np.argmin(np.sqrt(np.sum((current_input_vector-self.network)**2, axis=2))), (self.network_dimensions[0], self.network_dimensions[1]))

    def train(self, max_iterations):
        return
