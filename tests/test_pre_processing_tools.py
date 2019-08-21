import unittest
import src.tools.pre_processing_tools as ppt
import skimage.io as io
import os
from src.tools.helper_functions import timeit

import manage


class TestYoloTools(unittest.TestCase):
    """
    For testing the yolo tools

    """

    def setUp(self):
        """

        :return:
        """
        manage.log.debug("Setting up YoloTools tests")

        # Test images

        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'exampleImages', 'testingImages'))

        self.big_img = os.path.join(data_dir, 'CAM43_2014.jpg')
        self.list_small_img = [os.path.join(data_dir, '1_CAM43_2014.jpg'), os.path.join(data_dir, '2_CAM43_2014.jpg'),
                               os.path.join(data_dir, '3_CAM43_2014.jpg'), os.path.join(data_dir, '4_CAM43_2014.jpg'),
                               os.path.join(data_dir, '5_CAM43_2014.jpg')]

        self.ppt_yolo = ppt.YoloTools()

    def test_template_match(self):
        """
        We know the location of the first image in the template

        :return:
        """
        x, y, w, h = self.ppt_yolo.template_match(self.list_small_img[0], self.big_img, normalised_coords=False) #, image_num=0)
        print('{} {}'.format(x, y))
        self.assertEquals(x, 646)
        self.assertEquals(y, 293)

    @timeit
    def test_template_batch_match(self):
        """
        We check against a batch of images

        :return:
        """
        _x = [646, 819, 521, 1376, 256]
        _y = [293, 401, 818, 519, 608]

        for ii, small_img in enumerate(self.list_small_img):
            x, y, w, h = self.ppt_yolo.template_match(small_img, self.big_img, normalised_coords=False)
            print('{} {}'.format(x, y))

            self.assertEquals(x, _x[ii])
            self.assertEquals(y, _y[ii])

    @timeit
    def test_par_template_batch_match(self):
        """
        Test the parallel implementation
        :return:
        """
        _x = [646, 819, 521, 1376, 256]
        _y = [293, 401, 818, 519, 608]
        self.ppt_yolo.par_template_match(self.list_small_img, self.big_img, normalised_coords=False)

    @timeit
    def test_generate_yolo_labels(self):
        """

        :return:
        """
        print('Generating yolo files')
        for ii, small_img in enumerate(self.list_small_img):
            x, y, w, h = self.ppt_yolo.template_match(small_img, self.big_img)
            self.ppt_yolo.generate_yolo_labels(self.big_img, '0', [x, y, w, h])
