import os
import xml.etree.ElementTree as ET
import imagesize
import re

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

def write_darknet(labelled_image, label_list):
    filename, _ = os.path.splitext(labelled_image.path)
    output_path = os.path.abspath(filename + '.txt')

    with open(output_path, 'w') as output_file:
        for i, obj in enumerate(labelled_image.objects):
            if i > 0:
                output_file.write('\n')

            scaled_cwh = obj.to_scaled_cwh(labelled_image.size)
            output_file.write('%d %.6f %.6f %.6f %.6f' % (label_list.index(obj.label), *scaled_cwh))

    return output_path

def write_voc(labelled_image, output_dir):
    object_xml = ''

    for obj in labelled_image.objects:
        object_xml += _VOC_ANNOTATION_OBJECT_FORMAT.format(
            label=obj.label, 
            xmin=obj.box[0], 
            ymin=obj.box[1],
            xmax=obj.box[2],
            ymax=obj.box[3]
        )

    path = os.path.abspath(labelled_image.path)
    filename, ext = os.path.splitext(os.path.basename(path))
    output_path = os.path.join(output_dir, filename + '.xml')

    with open(output_path, 'w') as output_file:
        voc_xml = _VOC_ANNOTATION_FORMAT.format(
            folder = os.path.dirname(path),
            filename = filename + ext,
            path = path,
            width = labelled_image.size[0],
            height = labelled_image.size[1],
            object_xml = object_xml
        )

        output_file.write(voc_xml)

    return output_path
    
def read_voc(voc_path):
    voc_xml = ET.parse(voc_path)
    size_xml = voc_xml.find('size')
    image_path = voc_xml.find('path').text
    objects = []

    for object_xml in voc_xml.findall('object'):
        label = object_xml.find('name').text
        box_xml = object_xml.find('bndbox')

        objects.append(LabelledObject(label, (
            float(box_xml.find('xmin').text),
            float(box_xml.find('ymin').text),
            float(box_xml.find('xmax').text),
            float(box_xml.find('ymax').text) 
        )))

    size = (float(size_xml.find('width').text), float(size_xml.find('height').text))
    return LabelledImage(image_path, size, objects)

def read_darknet(image_path, label_list):
    # note that the image_path is passed in, not the path to the annotation
    # we can derive the annotation file path from the image path but not vice-versa
    # as the image could have any extension (.png/.jpg/.tiff/etc.)
    filename, _ = os.path.splitext(image_path)
    size = imagesize.get(image_path)
    objects = []

    with open(filename + '.txt', 'r') as darknet_file:
        for line in darknet_file:
            parts = re.split('\s+', line.rstrip())
            scx, scy, sw, sh = map(float, parts[1:5])
            label = label_list[int(parts[0])]
            
            # scale the box back to actual pixel coords
            w = sw * size[0]
            h = sh * size[1]
            x = (scx * size[0]) - (w / 2)
            y = (scy * size[1]) - (h / 2)

            objects.append(LabelledObject(label, (x, y, x + w, y + h)))

    return LabelledImage(image_path, size, objects)
