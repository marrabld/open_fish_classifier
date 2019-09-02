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
import pandas
import traceback

import skimage.measure

from PIL import Image

class Extract:

    def __init__(self):
        return

    def find_fish_bounds(self, lx0, ly0, lx1, ly1):

        original_lx0 = lx0
        original_lx1 = lx1
        original_ly0 = ly0
        original_ly1 = ly1

        # if fish is measured right to left, we need to swap the xys so we can draw a bounding box consistently
        lx0 = original_lx0 if original_lx0 < original_lx1 else original_lx1
        lx1 = original_lx0 if original_lx0 > original_lx1 else original_lx1
        ly0 = original_ly0 if original_ly0 < original_ly1 else original_ly1
        ly1 = original_ly0 if original_ly0 > original_ly1 else original_ly1
        midy = ly0 + ((ly1 - ly0) / 2)
        diffy = ly1 - ly0

        # adjust the xys to crop a bounding box
        width = lx1 - lx0
        quarter_width = width / 3

        ly0_tmp = midy - quarter_width
        ly1_tmp = midy + quarter_width

        if ((diffy) < (ly1_tmp - ly0_tmp)):
            ly0 = midy - quarter_width
            ly1 = midy + quarter_width

        if ly0 < 0:
            ly0 = 0

        #if lx0 >= lx1 or ly0 >= ly1:
        #    print("Adju", lx0, ly0, lx1, ly1)
        #    print("False")
        #    # exit(0)

        return int(lx0), int(ly0), int(lx1), int(ly1)

    def crop_and_save_fish(self, image, box, destination_path):
        b = np.array(box).astype(int)
        fish = image.crop((b[0], b[1], b[2], b[3]))
        fish.save(destination_path, "PNG")

    def yes(question):
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        return False

    def check_video_files_exist(self, video_base_directory, filenamelist):
        for filename in filenamelist:
            if not os.path.exists(os.path.join(video_base_directory, filename)):
                print("Warning", filename, "does not exist will have to skip.")

    def measurements_to_bounding_boxes(self,
                                       video_base_directory,
                                       em_lengths_file,
                                       em_image_pt_pair_file,
                                       output_directory):

        # check output path exists
        if not os.path.exists(output_directory):
            print(output_directory, "does not exist.")

        # check that output dir is empty, if not then quit
        for dirpath, dirnames, files in os.walk(output_directory):
            if files:
                print(dirpath, 'has files in it that could be overwritten. Choose a different path.')
                exit(0)

        frames_output_directory = os.path.join(output_directory, "Frames")
        crops_output_directory = os.path.join(output_directory, "Crops")

        try:
            # make frames path
            os.mkdir(frames_output_directory)
            os.mkdir(crops_output_directory)
        except OSError:
            print("Creation of the directory outputs failed.")
            traceback.print_exc()
            exit(0)

        df_em_lengths = pandas.read_csv(em_lengths_file, sep='\t', lineterminator='\r')
        df_em_pt_pair = pandas.read_csv(em_image_pt_pair_file, sep='\t', lineterminator='\r')

        df = pandas.merge(df_em_lengths, df_em_pt_pair, on=['OpCode', 'ImagePtPair'])
        df["Index"] = pandas.to_numeric(df["Index"])
        df["FrameLeft"] = pandas.to_numeric(df["FrameLeft"])
        df["FrameRight"] = pandas.to_numeric(df["FrameRight"])

        op_codes = df.OpCode.unique()
        point_pairs = df.ImagePtPair.unique()

        species_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])
        genus_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])
        family_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])

        species_crop_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])
        genus_crop_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])
        family_crop_df = pandas.DataFrame(columns=['filename', 'x0', 'y0', 'x1', 'y1', 'label'])

        left_videos = df.FilenameLeft.unique()
        right_videos = df.FilenameRight.unique()

        left_videos = [x for x in left_videos if str(x) != 'nan']
        right_videos = [x for x in right_videos if str(x) != 'nan']

        self.check_video_files_exist(video_base_directory, left_videos)
        self.check_video_files_exist(video_base_directory, right_videos)

        count = 0
        for code in op_codes:
            for point in point_pairs:

                if df[(df.OpCode == code) & (df.ImagePtPair == point)].shape[0] == 0:
                    continue

                ######################################## do left frame

                filename_left = df[(df.OpCode == code) & (df.ImagePtPair == point)]["FilenameLeft"].values[0]
                frame_left = df[(df.OpCode == code) & (df.ImagePtPair == point)]["FrameLeft"].values[0]
                lx0 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 0)]["Lx"].values[0]
                ly0 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 0)]["Ly"].values[0]
                lx1 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 1)]["Lx"].values[0]
                ly1 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 1)]["Ly"].values[0]

                frame_left = int(frame_left)

                species = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Species"].values[0]
                genus = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Genus"].values[0]
                family = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Family"].values[0]

                lx0, ly0, lx1, ly1 = self.find_fish_bounds(lx0, ly0, lx1, ly1)

                if lx0 >= lx1 or ly0 >= ly1:
                    print("skipping", filename_left, frame_left, lx0, ly0, lx1, ly1, family, genus, species)
                    continue

                video = os.path.join(video_base_directory, filename_left)

                if not os.path.exists(video):
                    print("continuing")
                    continue

                output_frame_path = os.path.join(frames_output_directory, filename_left + "." + str(int(frame_left)) + ".png")
                fish_output_frame_path = os.path.join(crops_output_directory,
                                                      filename_left + "." + str(int(frame_left)) + "." + str(
                                                          int(lx0)) + "." + str(int(ly0)) + "." + str(
                                                          int(lx1)) + "." + str(int(ly1)) + ".png")

                species_df = species_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": species},
                    ignore_index=True)
                genus_df = genus_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": genus},
                    ignore_index=True)
                family_df = family_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": family},
                    ignore_index=True)

                species_crop_df = species_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": species},
                    ignore_index=True)
                genus_crop_df = genus_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": genus},
                    ignore_index=True)
                family_crop_df = family_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": family},
                    ignore_index=True)

                cap = cv2.VideoCapture(video)  # video_name is the video being called
                # amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                # print(amount_of_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_left))  # Where frame_no is the frame you want
                ret, frame_left = cap.read()  # Read the frame

                pil_image = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(pil_image)

                # save frame
                pil_image.save(output_frame_path, "PNG")

                # save crop
                self.crop_and_save_fish(pil_image, (lx0, ly0, lx1, ly1), fish_output_frame_path)

                ############################# do right frame

                filename_left = df[(df.OpCode == code) & (df.ImagePtPair == point)]["FilenameRight"].values[0]
                frame_left = df[(df.OpCode == code) & (df.ImagePtPair == point)]["FrameRight"].values[0]
                lx0 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 0)]["Rx"].values[0]
                ly0 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 0)]["Ry"].values[0]
                lx1 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 1)]["Rx"].values[0]
                ly1 = df[(df.OpCode == code) & (df.ImagePtPair == point) & (df.Index == 1)]["Ry"].values[0]

                frame_left = int(frame_left)

                species = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Species"].values[0]
                genus = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Genus"].values[0]
                family = df[(df.OpCode == code) & (df.ImagePtPair == point)]["Family"].values[0]

                lx0, ly0, lx1, ly1 = self.find_fish_bounds(lx0, ly0, lx1, ly1)

                if lx0 >= lx1 or ly0 >= ly1:
                    print("skipping", filename_left, frame_left, lx0, ly0, lx1, ly1, family, genus, species)
                    continue

                video = os.path.join(video_base_directory, filename_left)

                if not os.path.exists(video):
                    print("continuing")
                    continue

                output_frame_path = os.path.join(frames_output_directory,
                                                 filename_left + "." + str(int(frame_left)) + ".png")
                fish_output_frame_path = os.path.join(crops_output_directory,
                                                      filename_left + "." + str(int(frame_left)) + "." + str(
                                                          int(lx0)) + "." + str(int(ly0)) + "." + str(
                                                          int(lx1)) + "." + str(int(ly1)) + ".png")

                species_df = species_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": species},
                    ignore_index=True)
                genus_df = genus_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": genus},
                    ignore_index=True)
                family_df = family_df.append(
                    {"filename": output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": family},
                    ignore_index=True)

                species_crop_df = species_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": species},
                    ignore_index=True)
                genus_crop_df = genus_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": genus},
                    ignore_index=True)
                family_crop_df = family_crop_df.append(
                    {"filename": fish_output_frame_path, "x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1, "label": family},
                    ignore_index=True)

                cap = cv2.VideoCapture(video)  # video_name is the video being called
                # amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                # print(amount_of_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_left))  # Where frame_no is the frame you want
                ret, frame_left = cap.read()  # Read the frame

                pil_image = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(pil_image)

                # save frame
                pil_image.save(output_frame_path, "PNG")

                # save crop
                self.crop_and_save_fish(pil_image, (lx0, ly0, lx1, ly1), fish_output_frame_path)

                break

        species_df.to_csv(os.path.join(output_directory, "SpeciesBBOXAnnotations-FromMeasurementFile.csv"))
        family_df.to_csv(os.path.join(output_directory, "FamilyBBOXAnnotations-FromMeasurementFile.csv"))
        genus_df.to_csv(os.path.join(output_directory, "GenusBBOXAnnotations-FromMeasurementFile.csv"))

        species_crop_df.to_csv(os.path.join(output_directory, "SpeciesCropAnnotations-FromMeasurementFile.csv"))
        family_crop_df.to_csv(os.path.join(output_directory, "FamilyCropAnnotations-FromMeasurementFile.csv"))
        genus_crop_df.to_csv(os.path.join(output_directory, "GenusCropAnnotations-FromMeasurementFile.csv"))

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

                            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(rgb_frame, '{} {}'.format(item['Genus'], item['Species']),
                                        (xpix + 50, ypix + 50), font, 2, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            cv2.circle(rgb_frame, (xpix, ypix), 2, (255, 255, 255), 2)

                            self.box_found[frame] = found_box_list

                            pylab.imshow(rgb_frame)
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

    def draw_all_annotations(self):
        """

        :return:
        """

        root = Tk()

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
                    nf = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                    for box in box_proposals[str(frame)]:
                        x, y, w, h = box
                        points = np.array([xpix, ypix])
                        verts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                        if True:
                            print('Found Box!! frame {}'.format(frame))
                            self.match_found = True
                            num_items += 1

                            # Keep track of the ones that we find.
                            found_box_list.append([x, y, w, h])

                            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(rgb_frame, '{} {}'.format(item['Genus'], item['Species']),
                                        (xpix + 50, ypix + 50), font, 2, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            cv2.circle(rgb_frame, (xpix, ypix), 2, (255, 255, 255), 2)

                            self.box_found[frame] = found_box_list


                            crop = nf[y:y + h, x:x + w]
                            cv2.imwrite(
                                "mov/proposal_crops/fish_crop_%s-%s-%s-%s.png" % (str(x), str(y), str(w), str(h)), crop)

                    pylab.imshow(rgb_frame)
                    pylab.savefig(
                        'mov/allproposals/box_{video}_{frame}_{num_item}.jpg'.format(
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

