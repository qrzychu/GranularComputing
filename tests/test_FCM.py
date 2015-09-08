from unittest import TestCase

import numpy as np

from fcm import FCM


class TestFCM(TestCase):
    def test_clusterize(self):
        fcm = FCM()

        for c in range(2, 5):
            n = 200
            d = 15
            x = np.random.random([n, d]) * 10

            v, u = fcm.clusterize(x, c)

            for i in range(n):
                self.assertAlmostEqual(1, sum(u[i]), delta=1e-5)

    def test__dist(self):
        fcm = FCM()
        x = np.array([0, 0, 0])
        y = np.array([0, 2, 0])

        self.assertEqual(2, fcm._dist(x, y))
        self.assertEqual(fcm._dist(x, y), fcm._dist(y, x))
