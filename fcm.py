import numpy as np


class FCM(object):
    """ Class wrapping Fuzzy C-Means algorithm """

    def __init__(self, m=2, e=1e-6):
        self.N = 0
        self.m = m
        self.e = e
        self.X = None
        self.N = None
        self.X = None
        self.d = None
        self.V = None
        self.D = None
        self.U = None

    def clusterize(self, X, C):
        """Runs clusterization of X into c fuzzy clusters

        returns ( c - clusters centres, U - membership matrix)
        """

        self.X = X
        self.N = len(self.X)
        self.X = np.array(self.X)
        self.d = len(X[0])
        self.V = np.zeros([C, self.d])

        # step 1: rand membership matrix
        self.U = np.random.rand(self.N, C)

        iter_no = 0
        # self.calc_distances()
        # step 2: update membership matrix U and clusters centres until U stops changing
        while True:

            last_u = np.copy(self.U)

            # update clusters centres as mean of all points weighted with
            for k in range(C):
                # for each cluster
                up = np.zeros([self.d, 1])
                down = np.zeros([self.d, 1])
                for i in range(self.d):
                    # for each data dimension
                    for x in range(self.N):
                        # for each element of dataset
                        up[i] += self.U[x, k] * self.X[x, i]
                        down[i] += self.X[x, i]

                # calc new cluster center
                self.V[k, :] = (up / down).flatten()

            # update membership matrix

            for i in range(self.N):
                # for each dataset element
                for k in range(C):
                    # for each cluster

                    # calc distance of i-th point to center of k-th cluster
                    dik = self._dist(self.X[i], self.V[k])

                    # update membership matrix
                    self.U[i, k] = 1.0 / sum(
                        [(dik / self._dist(self.X[i], self.V[j])) ** (2.0 / (self.m - 1)) for j in range(C)])

            iter_no += 1
            if np.max(abs(last_u - self.U)) <= self.e or iter_no > 1000:
                break

        return self.V, self.U

    @staticmethod
    def _dist(x, y):
        return np.sum(np.power(x - y, 2)) ** 0.5

    def calc_distances(self):
        self.D = np.zeros([self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.D[i, j] = np.sum(np.power(self.X[i] - self.X[j], 2)) ** 0.5

# tests
#
# fcm = FCM()
#
# x = [[1, 9, 13],
#      [2, 8, 23],
#      [3, 7, 33],
#      [4, 6, 43],
#      [5, 5, 53]]
#
# [v,u] = fcm.clusterize(x, 2)
#
# print("V:")
# print(v)
# print("U:")
# print(u)
