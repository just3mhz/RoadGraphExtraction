import numpy as np
import rtree as rt

from scipy.ndimage import gaussian_filter
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_erosion
from scipy.ndimage import maximum_filter
from scipy.spatial import distance


def _shifts(x, y, radius):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            yield x + dx, y + dy


class GraphTensorEncoder:
    def __init__(self, image_size, max_degree, d, pv_threshold=0.5, pe_threshold=0.5):
        self.image_size = image_size
        self.max_degree = max_degree
        self.d = d
        self.radius = 1
        self.pv_threshold = pv_threshold
        self.pe_threshold = pe_threshold
        self.w = 100

    def encode(self, graph):
        encoded = np.zeros((self.image_size, self.image_size, 3 * self.max_degree + 1))

        for (xv, yv) in graph.keys():

            for x, y in _shifts(xv, yv, self.radius):
                #  Probability of having vertex at position (x, y)
                encoded[x, y, 0] = 0.9

            encoded[xv, yv, 0] = 1.0

            for i, (xu, yu) in enumerate(graph[(xv, yv)]):
                dx = xu - xv
                dy = yu - yv

                angle = np.arctan2(dx, dy) * 180 / np.pi + 180  # degrees

                sector = int(angle / (360 / self.max_degree)) % self.max_degree

                for x, y in _shifts(xv, yv, self.radius):
                    #  Probability of having edge from position (x, y) to (x + dx, y + dy)
                    encoded[x, y, 1 + 3 * sector] = 1
                    #  Normalized dx
                    encoded[x, y, 2 + 3 * sector] = dx / self.d
                    #  Normalized dy
                    encoded[x, y, 3 + 3 * sector] = dy / self.d

        return encoded

    def decode(self, encoded):
        vertexes = self._detect_vertexes(gaussian_filter(encoded[:, :, 0], 1, mode='constant'))
        #  Insert vertexes in rtree to speed up queries
        rtree_index = rt.Index()
        for i, (xv, yv) in enumerate(vertexes):
            rtree_index.insert(i, [xv - 1, yv - 1, xv + 1, yv + 1])
        #  Connect vertexes
        graph = {(xv, yv): [] for (xv, yv) in vertexes}
        for i, (xv, yv) in enumerate(vertexes):
            for j in range(self.max_degree):
                pe = encoded[xv, yv, 3*j + 1]
                if pe < self.pe_threshold:
                    continue
                dx = int(encoded[xv, yv, 3*j + 2] * self.d)
                dy = int(encoded[xv, yv, 3*j + 3] * self.d)

                candidates = list(rtree_index.intersection([
                    xv + dx - self.d,
                    yv + dy - self.d,
                    xv + dx + self.d,
                    yv + dy + self.d
                ]))

                min_distance = np.inf
                best_candidate = None

                def dist(xu, yu):
                    v0 = np.array([xv + dx, yv + dy])
                    v1 = np.array([xu, yu])
                    v2 = np.array([xu - xv, yu - yv])
                    v3 = np.array([dx, dy])
                    return distance.euclidean(v0, v1) + self.w * distance.cosine(v2, v3)

                for c in candidates:
                    if c == i:
                        continue
                    current_distance = dist(*vertexes[c])

                    if current_distance < min_distance:
                        min_distance = current_distance
                        best_candidate = c

                if best_candidate is not None:
                    graph[xv, yv].append(vertexes[best_candidate])

        return graph

    def _detect_vertexes(self, vertex_channel):
        neighborhood = generate_binary_structure(len(vertex_channel.shape), 2)

        local_max = (maximum_filter(vertex_channel, footprint=neighborhood) == vertex_channel)

        background = (vertex_channel == 0)

        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)

        detected_maxima = local_max ^ eroded_background
        x, y = np.where((detected_maxima & (vertex_channel > self.pv_threshold)))

        return [(xi, yi) for (xi, yi) in zip(x, y)]


if __name__ == '__main__':
    graph = {
        (1, 1): [(2, 8), (5, 3)],
        (2, 8): [(1, 1), (5, 7)],
        (5, 3): [(5, 7), (8, 1)],
        (5, 7): [(2, 8), (5, 3)],
        (8, 1): [(5, 3)]
    }

    encoder = GraphTensorEncoder(10, 6, 5)
    encoded = encoder.encode(graph)

    decoded = encoder.decode(encoded)

    print(graph)
    print(decoded)
