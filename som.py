import numpy as np


class SOM:

    def __init__(self, Size_X, Size_Y, dim_of_input_vec):

        self.network_dimensions = (Size_X, Size_Y, dim_of_input_vec)
        self.network = np.random.random(self.network_dimensions)
        return

    def train(self, max_iterations):
        return


if __name__ == '__main__':
    n = SOM(4, 4, 10)
    print(n.network)
