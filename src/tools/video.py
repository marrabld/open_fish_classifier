import cv2
import numpy as np
import os
import glob


class RenderVideo:
    """Render video from a directory of jpgs"""

    def __init__(self, directory):
        """

        :param video:
        """

        self.directory = directory

    def render(self):
        """

        :return:
        """

        img_array = []

        for filename in glob.glob(os.path.join(self.directory, '*.jpg')):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


class RipVideo:
    """Break the video into jpg files for frame by frame analysis"""

    def __init__(self, video, start_frame=None, end_frame=None):
        """
        Constructor

        :param video:
        """
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame

    def render(self):
        """Turn each frame into a jpg"""

        cap = cv2.VideoCapture(self.video)

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        if not self.start_frame:
            f_num = 0
            while 1:
                ret, frame2 = cap.read()
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join('mov', 'rgb', '{:04d}.jpg'.format(f_num)), frame2)

                f_num += 1

                cv2.imshow('frame2', frame2)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png', frame2)

                prvs = next



        elif self.start_frame and self.end_frame:
            # We only want to grab frames of interest

            cap = cv2.VideoCapture(self.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            f_num = self.start_frame
            while True:
                ret, frame2 = cap.read()
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join('mov', 'rgb', '{:04d}.jpg'.format(f_num)), frame2)

                f_num += 1

                cv2.imshow('frame2', frame2)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                elif f_num == self.end_frame + 1:
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png', frame2)

                prvs = next

        cap.release()
        cv2.destroyAllWindows()
