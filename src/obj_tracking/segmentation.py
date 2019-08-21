import glob

import cv2
import numpy as np
import skimage.morphology
import os


class Motion:
    """
    Methods of segmenting fish from video based on their motion
    """

    def __init__(self, video: str, start_Frame=None, end_frame=None) -> None:
        """
        :video: path to video

        """
        self.video = video  # VIDEO_FILE = '/home/marrabld/Videos/vlc-record-2019-07-26-11h54m12s-224_L23.avi'
        self.method = 'LSBP' #'CNT' #'LSBP' #'GSOC' # 'LSBP'  # MOG, GSOC, CNT
        self.kernel = (15, 15)
        self.save_frame = True
        self.start_frame = start_Frame
        self.end_frame = end_frame

    def render(self) -> None:
        """
        The main entry point

        :return: None
        """

        f_num = 0  # start our frame counter
        cap = cv2.VideoCapture(self.video)

        # Used for dilating the segmented mask and 'fill holes'
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           self.kernel)  # Play with these numbers to be more or less aggressive
        if self.method == 'LSBP':
            fgbg = cv2.cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif self.method == 'MOG':
            fgbg = cv2.cv2.bgsegm.createBackgroundSubtractorMOG()
        elif self.method == 'GSOC':
            fgbg = cv2.cv2.bgsegm.createBackgroundSubtractorGSOC()
        elif self.method == 'CNT':
            fgbg = cv2.cv2.bgsegm.createBackgroundSubtractorCNT()
        else:
            raise Exception('Incorrect Subtraction Method')

        if not self.start_frame:
            while True:
                ret, frame = cap.read()
                if ret:
                    fgmask = fgbg.apply(frame)
                    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                    cv2.imshow('frame', fgmask)

                    if self.save_frame:
                        print('saving frame to {}'.format(os.path.join('mov', 'bak', '1', '{:04d}.jpg'.format(f_num))))
                        # This may limit the number of frames due to :04d  increase for more frames
                        cv2.imwrite(os.path.join('mov', 'bak', '1', '{:04d}.jpg'.format(f_num)), fgmask)
                        f_num += 1

                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    break

        elif self.start_frame and self.end_frame:
            cap = cv2.VideoCapture(self.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            f_num = self.start_frame
            while True:
                ret, frame = cap.read()
                if ret:
                    fgmask = fgbg.apply(frame)
                    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                    cv2.imshow('frame', fgmask)

                    if self.save_frame:
                        print('saving frame to {}'.format(os.path.join('mov', 'bak', '1', '{:04d}.jpg'.format(f_num))))
                        # This may limit the number of frames due to :04d  increase for more frames
                        cv2.imwrite(os.path.join('mov', 'bak', '1', '{:04d}.jpg'.format(f_num)), fgmask)
                        f_num += 1

                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                    elif f_num == self.end_frame + 1:
                        break
                else:
                    break

        cap.release()
        cv2.destroyAllWindows()


class IsolateObjects:
    """
    Use the background masks to isolate the moving objects. .. the fish!
    """

    def __init__(self, video=None):
        self.kernel = (13, 13)
        self.box_proposals = {}
        self.video = video

    def render(self) -> None:
        """
        Render the video
        :return: Save to mov/merge
        """

        # Load an color image in grayscale
        frame = 0
        rgb_file_list = [f for f in glob.glob('mov/rgb' + "**/*.jpg", recursive=False)]
        bak_1_file_list = [f for f in glob.glob('mov/bak/1' + "**/*.jpg", recursive=False)]

        while True:  ## Change me

            try:
                rgb_file = rgb_file_list.pop()
                bak_file = bak_1_file_list.pop()
            except:
                break

            frame = rgb_file.split('/')[-1].split('.')[0]  # Grab file frame from the filename

            print(frame)
            # frame += 1
            # print('Reading :: {:04d}.jpg'.format(frame))
            # img = cv2.imread('mov/hsv/{:04d}_hsv.png'.format(frame - 1), cv2.IMREAD_COLOR)
            img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            fg1 = cv2.imread(bak_file, cv2.IMREAD_GRAYSCALE)

            # fg2 = cv2.imread('mov/bak/2/{:04d}.png'.format(frame), cv2.IMREAD_GRAYSCALE)
            # fg2 = cv2.imread('mov/hsv/{:04d}_hsv.png'.format(frame -1), 0)

            fg = fg1  # + fg2#np.logical_and(fg1, fg2)

            # Try this or turn them off
            fg = skimage.morphology.dilation(fg)
            fg = skimage.morphology.closing(fg)

            kernel = np.ones(self.kernel, np.uint8)  ## 13,13
            # kernel = np.ones((13, 13), np.uint8)

            # fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

            contours, hierarchy = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # try:
            #     im3 = cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
            # except:
            #     pass
            # import pylab
            # pylab.imshow(im3)
            # pylab.show()

            # create hull array for convex hull points
            hull = []
            #
            # calculate points for each contour
            for i in range(len(contours)):
                # creating convex hull object for each contour
                hull.append(cv2.convexHull(contours[i], False))

            # create an empty black image
            drawing = np.zeros((fg.shape[0], fg.shape[1], 3), np.uint8)
            drawing1 = np.zeros((fg.shape[0], fg.shape[1], 3), np.uint8)
            drawing3 = np.zeros((fg.shape[0], fg.shape[1], 3), np.uint8)

            # draw contours and hull points
            for i in range(len(contours)):
                color_contours = (0, 255, 0)  # green - color for contours
                color = (255, 0, 0)  # blue - color for convex hull
                # draw ith contour
                fg = cv2.drawContours(drawing, contours, i, color_contours, -1)  # , 8, hierarchy, -1)
                # draw ith convex hull object
                fg_c = cv2.drawContours(drawing1, hull, i, color, -1)  # , 8, -1)

                box_list = []

                for cnt in hull:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(drawing3, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    box_list.append([x, y, w, h])

                self.box_proposals[frame] = box_list


                # kernel = np.ones((2, 2), np.uint8)
                # gradient = cv2.morphologyEx(drawing3, cv2.MORPH_GRADIENT, kernel)
                # drawing3 = gradient
                # box = cv2.boxPoints(rect)
                # fg = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                # boxes = []
                # for c in contours:
                #     (x, y, w, h) = cv2.boundingRect(c)
                # boxes.append([x, y, x + w, y + h])
                # boxes = np.asarray(boxes)
                # # need an extra "min/max" for contours outside the frame
                # left = np.min(boxes[:, 0])
                # top = np.min(boxes[:, 1])
                # right = np.max(boxes[:, 2])
                # bottom = np.max(boxes[:, 3])
                #
                # fg = cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            try:
                fg = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
                fg_c = cv2.cvtColor(fg_c, cv2.COLOR_RGB2GRAY)
                thresh, fg = cv2.threshold(fg, 50, 255, cv2.THRESH_BINARY)
                thresh, fg_c = cv2.threshold(fg_c, 50, 255, cv2.THRESH_BINARY)
                thresh, fg_b = cv2.threshold(drawing3, 50, 255, cv2.THRESH_BINARY)

                # fg = np.logical_and(fg, fg2)
                fg = np.logical_and(fg, fg_b)
            except:
                pass

            merge = np.zeros_like(img)
            for ii in range(0, 2):
                try:
                    merge[:, :, ii] = (img[:, :, ii] * fg)  # + drawing3
                except:
                    # pass
                    print('passing')
                    merge[:, :, ii] = img[:, :, ii]

            # merge = cv2.cvtColor(merge, cv2.COLOR_BGR2RGB)   ###  <--- Check this

            # merge = cv2.add(merge, drawing3)  # drawing 3 is the boxes.
            alpha = 0.8
            beta = (1.0 - alpha)
            merge = cv2.addWeighted(merge, alpha, drawing3, beta, 0.0)
            cv2.imwrite('mov/merge/{}.jpg'.format(frame), merge)
            # cv2.imwrite('mov/merge/box_{:04d}.png'.format(frame), )

        # cv2.imshow('image', merge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        import json
        json_file = json.dumps(self.box_proposals)  # note i gave it a different name
        with open("mov/merge/{}_box_proposals.json".format(os.path.basename(self.video)), 'w') as f:
            f.write(json_file)

        print('Done')
