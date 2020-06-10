import os
import xml.etree.ElementTree as ET
import imagesize
import re
import csv
import json
import utils

_VOC_ANNOTATION_FORMAT = """
<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{path}</path>
    <source><database>Unknown</database></source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {object_xml}
</annotation>
"""

_VOC_ANNOTATION_OBJECT_FORMAT = """
<object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <bndbox>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
    </bndbox>
</object>
"""

class LabelledImage(object):
    def __init__(self, image_path, image_size, objects):
        self.path = image_path
        self.size = image_size
        self.objects = objects

class LabelledObject(object):
    def __init__(self, label, tlbr_box):
        self.label = label
        self.box = tlbr_box

    def to_tlwh(self):
        x1, y1, x2, y2 = self.box
        return (x1, y1, x2 - x1, y2 - y1)

    def to_cwh(self):
        x1, y1, x2, y2 = self.box
        w = x2 - x1
        h = y2 - y1

        return (x1 + (w/2), y1 + (h/2), w, h)

    def to_scaled_cwh(self, image_size):
        im_w, im_h = image_size
        cwh = self.to_cwh()
        return (cwh[0] / im_w, cwh[1] / im_h, cwh[2] / im_w, cwh[3] / im_h)

def _normalize_header(header):
    return re.sub('[^\w]+', '_', header.lower())

def create_frame_object_labeller(species_list, catch_all_species):
    def labeller(region, _):
        # Skip any regions that have no label in their attributes
        # TODO: Why are these even in the dataset?
        if 'label' not in region['region_attributes']:
            return None

        # the species labelling in the metadata won't match precisely with the input
        # target species labels so try and map them together here
        search = re.sub(r'\s+', '_', region['region_attributes']['label']).lower()
        return next((s for s in species_list if s.lower().endswith(search)), catch_all_species)

    return labeller

def parse_frame_images(metadata_path, frame_directory, label_object, skip_missing=False):
    """
    Generate a list of LabelledImages from an AIMS-provided metadata object (from the '_via_img_metadata' node)
    """
    images = []
    metadata = None

    with open(metadata_path, 'r') as metadata_file:
        content = json.load(metadata_file)
        metadata = content['_via_img_metadata']
    
    for meta in metadata.values():
        objects = []
        image_path = os.path.join(frame_directory, meta['filename'])
        image_size = utils.get_image_size(image_path)

        if image_size is None and not skip_missing:
            raise utils.FriendlyError('unable to locate file "%s" in "%s"' % (meta['filename'], frame_directory))

        for region in meta['regions']:
            shape = region['shape_attributes']
            tlbr = shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height']
            label = label_object(region, meta)

            if label is not None:
                objects.append(LabelledObject(label, tlbr))

        # each metadata entry corresponds to a single image
        # it doesn't matter if the image contains no annotated objects
        images.append(LabelledImage(image_path, image_size, objects))

    return images

def parse_crop_images(csv_path, crop_directory, label_object, skip_missing=False):
    """
    Generate a list of LabelledImages from an AIMS-provided crop CSV file
    """
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        images = []

        # First, map the column names to their index for sanity
        columns = utils.read_normalized_csv_headers(csv_reader)

        col_filename = columns['file_name']
        col_family = columns['family']
        col_genus = columns['genus']
        col_species = columns['species']

        for row in csv_reader:
            image_path = os.path.join(crop_directory, row[col_filename])
            image_size = utils.get_image_size(image_path)
            objects = []

            if image_size is None and not skip_missing:
                raise utils.FriendlyError('unable to locate file "%s" in "%s"' % (row[col_filename], crop_directory))

            label = label_object((row[col_family], row[col_genus], row[col_species]))
            
            # crops comprise of a single labelled object, spanning the entire image
            if label is not None:
                objects.append(LabelledObject(label, (0, 0, *image_size)))

            images.append(LabelledImage(image_path, image_size, [crop]))  

        return images