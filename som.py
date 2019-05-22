import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
from itertools import cycle


class SOM:

    def __init__(self, Size_X, Size_Y, dim_of_input_vec):
        self.network_dimensions = (Size_X, Size_Y, dim_of_input_vec)
        self.network = np.random.random(self.network_dimensions)
        # Works https://stackoverflow.com/questions/44230312/fastest-way-to-create-numpy-2d-array-of-indices/44230705#44230705
        self.network_map = np.indices(
            (self.network_dimensions[0], self.network_dimensions[1])).transpose(1, 2, 0)
        self.sigma_0 = max(self.network_dimensions[:2])/2
        self.alpha_0 = 0.1
        return

    def _find_bmu(self, current_input_vector):
        return np.unravel_index(np.argmin(np.sqrt(np.sum((current_input_vector-self.network)**2, axis=2))), (self.network_dimensions[0], self.network_dimensions[1]))

    def _calc_neighbourhood_radius(self, current_iter_number, max_iterations):
        lamDa = max_iterations/np.log(self.sigma_0)
        return self.sigma_0*np.exp(-current_iter_number/lamDa)

    def _calc_learning_rate(self, current_iter_number, max_iterations):
        lamDa = max_iterations/np.log(self.sigma_0)
        return self.alpha_0*np.exp(-current_iter_number/lamDa)

    def _alpha_influence_matrix(self, bmu_index, current_iter_number, max_iterations):
        # 𝜃𝑡=exp(−𝑑2𝑖2𝜎2𝑡)
        sigma_t_square = self._calc_neighbourhood_radius(
            current_iter_number, max_iterations)**2
        alpha_t = self._calc_learning_rate(current_iter_number, max_iterations)
        return alpha_t*np.exp(np.sum((self.network_map-bmu_index)
                                     ** 2, axis=2)*(-1/(2*sigma_t_square)))

    def _mass_weight_update(self, alpha_influence_matrix, current_input_vector):
        # https://stackoverflow.com/questions/46930268/numpy-multiple-3d-array-with-a-2d-array
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
        # einsum is pure GOLD!!!!
        self.network = self.network + \
            np.einsum('ij,ijk->ijk', alpha_influence_matrix,
                      current_input_vector-self.network)

    def train(self, input_data, max_iterations):
        nb_training_instances = len(input_data)
        for current_iter_number in range(1, max_iterations+1):
            current_input_vector = input_data[current_iter_number %
                                              nb_training_instances]
            current_bmu_index = self._find_bmu(current_input_vector)
            current_alpha_influence_matrix = self._alpha_influence_matrix(
                current_bmu_index, current_iter_number, max_iterations)
            self._mass_weight_update(
                current_alpha_influence_matrix, current_input_vector)

            # def _euclidean_distance(self, input_vector, weight_vector):
            #     return np.sqrt(np.sum((input_vector-weight_vector)**2))

            # def __li_find_bmu(self, current_input_vector):
            #     # list_euc_distances[euclidean_dist] = network_idx
            #     # if two are more neurons have the same bmu then the one that was discovered last would be chosen
            #     list_euc_distances = {}
            #     for x in range(self.network_dimensions[0]):
            #         for y in range(self.network_dimensions[1]):
            #             current_euclidean_distance = (sum(
            #                 [(current_input_vector[i]-self.network[x][y][i])**2 for i in range(self.network_dimensions[-1])]))**0.5
            #             list_euc_distances[current_euclidean_distance] = (x, y)
            #     return list_euc_distances[min(list(list_euc_distances.keys()))]

            # def _multi_find_bmu(self, current_input_vector):
            #     p = Pool(cpu_count())
            #     func = partial(self._euclidean_distance, current_input_vector)
            #     list_euc_distances = p.map(func, self.network.reshape(
            #         self.network_dimensions[0]*self.network_dimensions[1], self.network_dimensions[-1]))
            #     p.close()
            #     p.join()
            #     return np.unravel_index(np.argmin(np.asarray(list_euc_distances)),
            #                             (self.network_dimensions[0], self.network_dimensions[1]))


if __name__ == '__main__':
    n = SOM(10, 10, 3)
    print(n.network)
    # n.network = np.array([[[0.94563752, 0.63257259, 0.62330799, 0.9169576, 0.52090539,
    #                         0.76014365, 0.88731566, 0.13766359, 0.81698364, 0.75517215],
    #                        [0.24959696, 0.47899349, 0.5387374, 0.79552207, 0.18777454,
    #                         0.54878939, 0.90042839, 0.63369542, 0.15566938, 0.70214216],
    #                        [0.47749477, 0.8105684, 0.50862339, 0.31116534, 0.76798637,
    #                         0.14483127, 0.68716664, 0.23677695, 0.37666888, 0.97358715],
    #                        [0.52090694, 0.98997292, 0.3800422, 0.13516949, 0.14128138,
    #                         0.99974006, 0.10915691, 0.92611551, 0.1012386, 0.76566419]],

    #                       [[0.05311923, 0.515692, 0.14815773, 0.66084162, 0.84171136,
    #                         0.22514159, 0.45514003, 0.69290843, 0.45791252, 0.36659011],
    #                        [0.66840277, 0.53611257, 0.61016599, 0.59475133, 0.0747659,
    #                         0.39225803, 0.27649682, 0.80554383, 0.82573597, 0.20130497],
    #                        [0.02994509, 0.72906677, 0.71636909, 0.47638662, 0.72937043,
    #                         0.00458586, 0.86748466, 0.66691706, 0.78296873, 0.96987615],
    #                        [0.61697275, 0.32098207, 0.01394225, 0.91580146, 0.26253597,
    #                         0.63662354, 0.39392653, 0.26426677, 0.98840959, 0.24214804]],

    #                       [[0.62206724, 0.38560245, 0.0104204, 0.50421991, 0.68714163,
    #                         0.33940491, 0.29667143, 0.56186478, 0.74234771, 0.92092679],
    #                        [0.08513857, 0.75970068, 0.73223857, 0.32252895, 0.67574178,
    #                         0.68573562, 0.01112588, 0.13131632, 0.97498913, 0.42047014],
    #                        [0.37326507, 0.99233088, 0.4803127, 0.96581526, 0.75003124,
    #                         0.67385742, 0.15015521, 0.22328924, 0.22524415, 0.68324932],
    #                        [0.71219049, 0.79118856, 0.79297463, 0.36587878, 0.06976508,
    #                         0.58177913, 0.19377822, 0.51733297, 0.32991622, 0.77559093]],

    #                       [[0.94214166, 0.45600318, 0.59348307, 0.19674101, 0.72944553,
    #                         0.61529721, 0.0203898, 0.24333486, 0.3738245, 0.78736727],
    #                        [0.28462795, 0.33038608, 0.16718026, 0.89658484, 0.72805617,
    #                           0.20269702, 0.45492994, 0.58886186, 0.60998261, 0.17088489],
    #                        [0.71562685, 0.80982502, 0.59640266, 0.9196053, 0.93040441,
    #                           0.93716844, 0.12910673, 0.42044082, 0.38839653, 0.84327841],
    #                        [0.74080373, 0.25973613, 0.37891361, 0.30476513, 0.2986006,
    #                           0.45697623, 0.30357062, 0.15918931, 0.47555831, 0.50558186]]])
    # import time
    # # start = time.time()
    # # n._li_find_bmu(np.array([0.32634494, 0.31266583, 0.30478276, 0.65890959, 0.08491201,
    # #                       0.48429509, 0.57543958, 0.79627358, 0.53857862, 0.30190754]))
    # # print(time.time()-start)
    # c = np.array([0.32634494, 0.31266583, 0.30478276, 0.65890959, 0.08491201,
    #               0.48429509, 0.57543958, 0.79627358, 0.53857862, 0.30190754])
    # np_start = time.time()
    # bmu_index = n._find_bmu(c)
    # print(time.time()-np_start)

    # # p_start = time.time()
    # # (n._multi_find_bmu(np.array([0.32634494, 0.31266583, 0.30478276, 0.65890959, 0.08491201,
    # #                              0.48429509, 0.57543958, 0.79627358, 0.53857862, 0.30190754])))
    # # print(time.time()-p_start)
    # alpha_m = n._alpha_influence_matrix(bmu_index, 10, 100)
    # print(alpha_m)

    # new_net = n._mass_weight_update(alpha_m, c)

    # print(new_net)

    input_data = np.array([[0.06623834,  0.41323115,  0.5055075],
                           [0.27841281,  0.35846549,  0.46476453],
                           [0.06462256,  0.63723597,  0.33711911],
                           [0.56441382,  0.09053535,  0.55779798],
                           [0.65837748,  0.95236121,  0.17537656],
                           [0.79862749,  0.81493842,  0.56016761],
                           [0.31951876,  0.67071923,  0.1332033],
                           [0.51785355,  0.93340037,  0.48103237],
                           [0.5538335,  0.35650457,  0.00457656],
                           [0.90240766,  0.95776868,  0.39686523]])

    n.train(input_data, 100)

    import matplotlib.pyplot as plt

    plt.show(n.network)
