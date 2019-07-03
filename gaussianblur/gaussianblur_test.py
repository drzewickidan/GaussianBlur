import hw1
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


class Homework1Test(unittest.TestCase):

    def test_create_gaussian_kernel(self):
        result = hw1.create_gaussian_kernel(5, 2.0)
        expected_result = np.array(
            [[0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
             [0.03382395, 0.04921355, 0.05576627, 0.04921355, 0.03382395],
             [0.03832756, 0.05576627, 0.06319146, 0.05576627, 0.03832756],
             [0.03382395, 0.04921355, 0.05576627, 0.04921355, 0.03382395],
             [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684]],
            dtype=np.float32
        )

        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(result.sum(), 1.0, places=4,
                               msg='The kernel sums to 1')
        difference = np.linalg.norm(result - expected_result)
        self.assertAlmostEqual(
            difference, 0.0, places=4, msg='The values for the kernel were not properly computed.')

    def test_convolve_pixel(self):
        kernel = np.array(
            [[0.1, 0.5, 0.9],
             [0.6, 0.3, 0.1],
             [0.1, 0.3, 0.8]],
            dtype=np.float32
        )
        image = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            dtype=np.uint8
        )
        resultOutOfBound = hw1.convolve_pixel(image, kernel, 2, 2)
        self.assertEqual(
            resultOutOfBound, 9, msg='Input pixel was not returned when the kernel was on the border.')
        result = hw1.convolve_pixel(image, kernel, 1, 1)
        print result
        self.assertAlmostEqual(result, 18.4, places=4,
                               msg='The corrent result is computed.')

    def test_convolve(self):
        kernel = np.array(
            [[0.11107408, 0.11112962, 0.11107408],
             [0.11112962, 0.11118521, 0.11112962],
             [0.11107408, 0.11112962, 0.11107408]],
            dtype=np.float32
        )
        image = np.ones((10, 10), dtype=np.uint8)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[0]):
                image[i, j] = i + j * 2

        result = hw1.convolve(image, kernel)
        expected_result = np.array(
            [[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
             [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19],
             [ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20],
             [ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21],
             [ 4,  6,  8, 10, 12, 14, 16, 18, 20, 22],
             [ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23],
             [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24],
             [ 7,  9, 11, 13, 15, 17, 19, 21, 23, 25],
             [ 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
             [ 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]],
            dtype=np.uint8)

        self.assertEqual(result.dtype, image.dtype)
        difference = np.linalg.norm(result - expected_result)
        self.assertAlmostEqual(
            difference, 0.0, msg='The values for the image were not properly computed.')

    def test_split_image(self):
        # Make a test color image. It should be 3 channels.
        color_image = np.ones((10, 10, 3), dtype=np.uint8)
        # Just make the pixel values average to the row number + 1
        for i in range(color_image.shape[0]):
            for c in range(0, 3):
                color_image[i, :, c] = i + c
        # split the image
        (r, g, b) = hw1.split(color_image)

        # ensure 2d, 1 channel images are returned
        self.assertEqual(r.shape, (10, 10))
        self.assertEqual(g.shape, (10, 10))
        self.assertEqual(b.shape, (10, 10))
        # ensure the types match
        self.assertEqual(r.dtype, color_image.dtype)
        self.assertEqual(g.dtype, color_image.dtype)
        self.assertEqual(b.dtype, color_image.dtype)

        # ensure the proper values are returned
        rDiff = np.linalg.norm(r - color_image[:, :, 0])
        gDiff = np.linalg.norm(g - color_image[:, :, 1])
        bDiff = np.linalg.norm(b - color_image[:, :, 2])
        self.assertAlmostEqual(
            rDiff, 0.0, msg='The first channel pixel values were not properly computed.')
        self.assertAlmostEqual(
            gDiff, 0.0, msg='The second channel pixel values were not properly computed.')
        self.assertAlmostEqual(
            bDiff, 0.0, msg='The third channel pixel values were not properly computed.')

    def test_merge_image(self):

        # create some test channels
        r = np.zeros((10, 10), dtype=np.uint8)
        g = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.uint8) * 2

        result = hw1.merge(r, g, b)

        # ensure the correct size & type is returned
        self.assertEqual(result.shape, (10, 10, 3))
        self.assertEqual(result.dtype, np.uint8)

        # ensure the proper values are returned
        self.assertAlmostEqual(result[:, :, 0].mean(
        ), 0.0, msg='The first channel pixel values were not properly computed.')
        self.assertAlmostEqual(result[:, :, 1].mean(
        ), 1.0, msg='The second channel pixel values were not properly computed.')
        self.assertAlmostEqual(result[:, :, 2].mean(
        ), 2.0, msg='The third channel pixel values were not properly computed.')


if __name__ == '__main__':
    unittest.main()
