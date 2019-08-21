import numpy as np
from skimage import io
import cv2 as cv
import os
from ..tools.meta import log as log
from ..tools.meta import config as config


class YoloTools():
    def __init__(self):
        """
        Tools for pre-processing the data ready for use in YOLO

        """
        pass

    def template_match(self, image_file, template_file, normalised_coords=False, draw_images=True, draw_box=False,
                       image_num=None,
                       threshold=0.99):
        """

        :param normalised_coords: True is required for YOLO files.   False for drawing the boxes.
        :param draw_images:
        :param template_file: The larger image or template we are finding image in.
        :param image_file: image we wish to locate
        :param image_num: for batch processing. add the image number to the file name
        :param threshold:  This is the search threshold.  Lower means more permissive.
        :return: tuple (x, y, w, h) coordinate of the x, y center of the image in the template and the w width and h height
        """
        if not image_num:
            image_num = ''

        image = io.imread(image_file)
        template = io.imread(template_file)

        ## DEBUG
        print(image_file)
        print(template_file)

        img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        tpl_rgb = template
        tpl = cv.cvtColor(template, cv.COLOR_RGB2GRAY)

        im = np.atleast_3d(image)
        tpl = np.atleast_3d(tpl)
        h, w, d = im.shape[:3]
        H, W, D = tpl.shape[:3]

        res = cv.matchTemplate(img_gray, tpl, cv.TM_CCOEFF_NORMED) #cv.TM_SQDIFF_NORMED)
        log.debug('{} confident we found a match'.format(np.max(res)))
        #threshold = np.max(res)
        loc = np.where(res >= np.float(threshold))
        # loc = np.min(res)#np.where(res >= threshold)

        if not loc:
            x, y, w, h = -1
            return x, y, w, h

        for pt in zip(*loc[::-1]):
            if draw_box:
                cv.rectangle(tpl_rgb, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), thickness=1)  # , 0.2)
                cv.circle(tpl_rgb, (pt[0] + int(w / 2), pt[1] + int(h / 2)), 1, (255, 255, 255), thickness=2)

            tmp = np.copy(tpl_rgb)

            if draw_images:
                tmp[:, :, 0] = tpl_rgb[:, :, 2]
                tmp[:, :, 1] = tpl_rgb[:, :, 1]
                tmp[:, :, 2] = tpl_rgb[:, :, 0]

                test_dir = config.get('MODEL', 'test_directory')
                file_name = image_num + '_' + os.path.split(image_file)[1].split('.')[0] + '_loc.png'
                file_name = os.path.join(test_dir, file_name)

                cv.imwrite(file_name, tmp)
            if normalised_coords:
                print((pt[0] + w) / 2.0 / W)

                #return (pt[0] + w) / 2.0 / W, (pt[1] + h) / 2.0 / H, np.float(w) / W, np.float(h) / H
                return (pt[0] + (0.5 * w) )  / W, (pt[1] + (0.5 * h)) / H, np.float(w) / W, np.float(h) / H

            return pt[0] + w / 2, pt[1] + h / 2, w, h

    def par_template_match(self, list_image_file, template_file, draw_images=False, image_num=None,
                           normalised_coords=True, threshold=0.99):
        """
        Parallel implementation of template match

        :param draw_images:
        :param template_file: The larger image or template we are finding image in.
        :param list_image_file: image we wish to locate
        :return: tuple (x, y, w, h) coordinate of the x, y center of the image in the template and the w width and h height
        :return:
        """

        import _thread as thread

        def unwrap_fun(image_file, template_file):
            obj = YoloTools()
            return obj.template_match(image_file, template_file, draw_images=draw_images, image_num=None,
                                      normalised_coords=normalised_coords, threshold=threshold)

        t = 0
        for image_file in list_image_file:
            print('Thread :: {}'.format(t))
            x, y, w, h = thread.start_new_thread(unwrap_fun, (image_file, template_file,))
            t += 1

        return x, y, w, h

    def directory_template_match(self, image_dir, template_dir, draw_box, normalised_coords):
        """

        :param image_dir:
        :param label_dir:
        :param draw_box:
        :param normalised_coords:
        :return:
        """

        images = os.listdir(image_dir)
        templates = os.listdir(template_dir)
        label_dir = config.get('MODEL', 'label_directory')
        threshold = config.get('PRE_PROCESSING_TOOL', 'template_threshold')

        k = 0

        for ii, image in enumerate(images):
            for jj, template in enumerate(templates):

                _image = os.path.join(image_dir, image)
                _template = os.path.join(template_dir, template)
                k += 1
                iter_num = k

                log.debug('Itter {} :: template {} :: fish {} :: Trying to find {} in {}'.format(iter_num, jj, ii, _image, _template))
                try:
                    x, y, w, h = self.template_match(_image, _template, normalised_coords=normalised_coords, draw_images=False,
                                                     draw_box=False, image_num=ii, threshold=threshold)
                    label_file = os.path.split(_template)[1].split('.')[0] + '.txt'
                    label_file = os.path.join(label_dir, label_file)
                    label = image_dir.split('/')[-1]
                    self.generate_yolo_labels(label_file, label, [x, y, w, h])
                    log.debug('x_pos :: {} , y_pos :: {}'.format(x, y))

                    #  We assume that we have one fish in one template.  therefore we can pop the fish and the template.
                    # This might results in errors if we find false positives.
                    # templates.pop(jj)
                    # images.pop(ii)
                    #break
                except:
                    log.debug('image {} not found in {}'.format(_image, _template))

    def streaming_template_match(self, image_dir, label_dir, movie_file, draw_box, normalised_coords):
        """
        This is for matching the images with a mppeg encoded video.  It requires you have to have ffmpeg installed.
        :return:
        """
        import subprocess

        template_threshold = config.get('PRE_PROCESSING_TOOL', 'template_threshold')
        log.debug('template threshold is :: {}'.format(template_threshold))

        # Step through all of the frames in the video and do a template match

        list_small_img = os.listdir(image_dir)

        frame_num = 0
        num_frames = 50 * 60 * 60
        num_seq = 1

        for frame_num in range(num_frames):
            print(frame_num)
            template = 'template.jpg'
            template = str(frame_num) + '_' + template
            call_string = 'ffmpeg -i {} -vf "select=eq(n\,{})" -vframes {} {}'.format(movie_file, frame_num, num_seq,
                                                                                      template)
            print(call_string)
            subprocess.call(call_string, shell=True)
            for ii, small_img in enumerate(list_small_img):
                small_img = os.path.join(image_dir, small_img)
                try:
                    log.info('Looking for {} in template'.format(small_img))
                    x, y, w, h = self.template_match(small_img, template, normalised_coords=normalised_coords,
                                                     draw_images=True, draw_box=draw_box,
                                                     image_num=str(frame_num), threshold=template_threshold)
                    label_file = str(frame_num) + '_' + os.path.split(small_img)[1].split('.')[0] + '_loc.txt'
                    label_file = os.path.join(label_dir, label_file)
                    self.generate_yolo_labels(label_file, '0', [x, y, w, h])
                    log.debug('x_pos :: {} , y_pos :: {}'.format(x, y))
                except:
                    log.exception("Can't find {}".format(list_small_img[ii]))

            os.remove(template)

    def generate_yolo_labels(self, image_file, label, coords):
        """

        :param label:
        :param image_file:
        :param coords:
        :return:
        """
        label_file = image_file.replace('.jpg', '.txt')

        with open(label_file, 'a') as f:
            f.write("{} {} {} {} {}".format(label, coords[0], coords[1], coords[2], coords[3]))
            f.write('\n')

    def generate_yolo_labels_from_bbtool(self, label_dir, out_dir, cat_number, Width=1920, Height=1080):
        """
        Given a driecotry of labels using teh BBtool.  Convert them into the format that YOLO needs

        :param out_dir:
        :param cat_number:
        :param label_dir:
        :return:
        """

        file_list = [file_list for file_list in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, file_list))]
        for file in file_list:
            with open(os.path.join(label_dir, file), 'r') as f:
                coords = f.readlines()

            for ii, coord in enumerate(coords):
                if ii == 0:  # skip the first line
                    pass
                else:
                    with open(os.path.join(out_dir, file), 'a') as fw:
                        # ============================== #
                        # convert the coordinates
                        # ============================== #
                        #    From
                        #    +---b---+
                        #    |       |
                        #    a       c
                        #    |       |
                        #    +---d---+
                        #   To
                        #   +-------+   ^
                        #   |       |   |
                        #   | (x,y) |   h
                        #   |       |   |
                        #   +-------+   |
                        #   ----w--->
                        coord = coord.split(' ')
                        a = np.float(coord[0])
                        b = np.float(coord[1])
                        c = np.float(coord[2])
                        d = np.float(coord[3])

                        w = (c - a)
                        h = (d - b)
                        x = a + (w / 2.0)
                        y = b + (h / 2.0)

                        fw.write("{} {} {} {} {}".format(cat_number, x / Width, y / Height, w / Width, h / Height))
                        fw.write('\n')


    def split_training_and_validation(self, label_directory, percentage_split=10):
        """
        Given a directory full of lables.  create two text files that define the text and validation data sets.

        :param percentage_split: 0 -> 1: 1 = 100%
        :param label_directory: label where all the yolo formated label files are
        :return:
        """

        import glob, os

        # Create and/or truncate train.txt and test.txt
        file_train = open('train.txt', 'w')
        file_test = open('test.txt', 'w')

        label_directory = os.path.abspath(label_directory)

        test_directory = label_directory.replace('labels', 'test')

        # Populate train.txt and test.txt
        counter = 1
        index_test = round(100.0 / percentage_split)
        for pathAndFilename in glob.iglob(os.path.join(label_directory, "*.txt")):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))

            if counter == index_test:
                counter = 1
                file_test.write(os.path.join(test_directory,  title) + '.jpg' + "\n")
            else:
                file_train.write(os.path.join(test_directory, title) + '.jpg' + "\n")
                counter = counter + 1