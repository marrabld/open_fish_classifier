import csv
from tkinter import filedialog
from tkinter import *
import os
import numpy as np
import cv2
import pylab
import json
from os.path import expanduser
import os.path

import skimage.measure


class Draw:
    """
    Drawing tools for overlaying event measure data onto video frames
    """

    def __init__(self, video=None, em_file=None, match_file=None):
        """
        Class constructor
        """

        self.try_match = True
        self.match_found = False
        self.video = video
        self.match_file = match_file
        self.em_file = em_file
        self.box_found = {}

    def draw_annotations(self):
        """

        :return:
        """

        root = Tk()
        # root.filename = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
        #                                            title="Select Event Measure file",
        #                                            filetypes=(("Event measure files", "*.txt"), ("all files", "*.*")))

        if self.em_file:
            root.filename = self.em_file
        else:
            home = expanduser("~")
            root.filename = filedialog.askopenfilename(initialdir=os.path.join(home, 'Videos'),
                                                       title="Select Event Measure file",
                                                       filetypes=(("Event measure files", "*.txt"), ("all files", "*.*")))
            print(root.filename)

        labels = []
        with open(root.filename) as f:
            reader = csv.DictReader(f, delimiter='\t')
            # dic = dict(reader)
            for row in reader:
                labels.append(row)

        if self.video:
            root.video_file = self.video
        else:
            root.video_file = filedialog.askopenfilename(initialdir=os.path.dirname(root.filename),
                                                         title="Select video file",
                                                         filetypes=(("Video Files", "*.*"), ("all files", "*.*")))

        #  251_L425_GP030009_3.MP4_box_proposals.json
        print('mov/merge/{video}_box_proposals.json'.format(video=os.path.basename(self.video)))
        # Don;t read this.  I am ashamed but i have an hour to go to get this working.
        if os.path.isfile('mov/merge/{video}_box_proposals.json'.format(video=os.path.basename(self.video))):
            self.match_file = 'mov/merge/{video}_box_proposals.json'.format(video=os.path.basename(self.video))

        else:
            self.match_file = None

        if self.match_file:
            box_file = self.match_file
            print('Found {} '.format(box_file))
        else:
            if self.try_match:
                box_file = filedialog.askopenfilename(initialdir='mov/merge',
                                                      title='Region Proposals',
                                                      filetypes=(("JSON Files", "*.json"), ("all files", "*.*")))
        if self.try_match:
            # read in our region proposals
            with open(box_file, 'r') as f:
                box_proposals = json.load(f)

        for item in labels:
            found_box_list = []
            drawing = np.zeros((1080, 1920, 3), np.uint8)  # todo get rid of magic numbers.
            xpix = np.int(np.float(item['ImageCol']))
            ypix = np.int(np.float(item['ImageRow']))
            frame = np.int(item['Frame'])
            annotation_video = item['Filename']

            start_frame_number = frame
            cap = cv2.VideoCapture(root.video_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

            # img = cv2.circle(drawing, (xpix, ypix), 10, (255, 255, 255), 1)

            if self.try_match and (annotation_video == os.path.basename(
                    self.video)):  # some Eventmeasure files reference multiple vids
                ret, rgb_frame = cap.read()
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                num_items = 0
                try:
                    for box in box_proposals[str(frame)]:
                        x, y, w, h = box
                        points = np.array([xpix, ypix])
                        verts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                        if skimage.measure.points_in_poly([points], verts):
                            print('Found Box!! frame {}'.format(frame))
                            self.match_found = True
                            num_items += 1

                            # Keep track of the ones that we find.
                            found_box_list.append([x, y, w, h])

                            # cv2.rectangle(drawing, (x, y), (x + w, y + h), (255, 255, 0), 1)
                            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                            # Debugging the box drawing
                            # dbg = np.zeros_like(drawing)
                            # cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 255, 0), 12)
                            # pylab.imshow(dbg)
                            # pylab.show()

                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(rgb_frame, '{} {}'.format(item['Genus'], item['Species']),
                                        (xpix + 50, ypix + 50), font, 2, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            cv2.circle(rgb_frame, (xpix, ypix), 2, (255, 255, 255), 2)

                            self.box_found[frame] = found_box_list

                            pylab.imshow(rgb_frame)
                            # pylab.show()
                            pylab.savefig(
                                'mov/found/box_{video}_{frame}_{num_item}.jpg'.format(
                                    video=os.path.basename(self.video),
                                    frame=frame, num_item=num_items),
                                dpi=300)
                            pylab.close()


                except:
                    print('No match found for {}'.format(frame))
                    self.match_found = False

            # grab the corresponding image from the video
            # ret, frame2 = cap.read()
            # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # try:
            #     img = img #+ frame2
            # except:
            #     pass

            # if self.match_found:
            #     pylab.imshow(rgb_frame)
            #     # pylab.show()
            #     pylab.savefig(
            #         'mov/found/box_{video}_{frame}.jpg'.format(video=os.path.basename(self.video), frame=frame),
            #         dpi=300)
            #     pylab.close()

            # mng = pylab.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())

        json_file = json.dumps(self.box_found)  # note i gave it a different name
        with open("mov/found/{}_box_found.json".format(os.path.basename(self.video)), 'w') as f:
            f.write(json_file)
