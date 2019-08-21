from manager import Manager
import os

from src.tools.video import RipVideo, RenderVideo
from src.tools.pre_processing_tools import YoloTools
from src.tools.meta import log as log
from src.tools.meta import config as config
from src.tools.event_measure import Draw

from src.obj_tracking.segmentation import Motion, IsolateObjects

environment = config.get('GLOB', 'environment')
log.info('Running in {} mode'.format(environment))

manager = Manager()


@manager.command
def run_tests():
    """Run the unit tests"""
    log.debug('Running Tests')
    os.system('nose2')


@manager.command
def run_draw_annotations():
    """Draw all of event measure annotations"""

    ann = Draw()
    ann.draw_annotations()


@manager.command
def run_segmentation(video, start_frame=None, end_frame=None):
    """Render the images without the background"""

    seg = Motion(video, int(start_frame), int(end_frame))
    seg.render()


@manager.command
def run_isolate_objects():
    """Merge the masks and the video together"""

    iso = IsolateObjects()
    iso.render()


@manager.command
def run_rip_video(video, start_frame=None, end_frame=None):
    """
    Turn each frame into a video.

    Example `python manage.py run_rip_video src/tools/14_Geo_224/224_L23.avi --start-frame 10 --end-frame 20`

    """

    rip = RipVideo(video, int(start_frame), int(end_frame))
    rip.render()

@manager.command
def run_rip_and_isolate(video, start_frame=None):
    """
    Meta script for now
    :return:
    """

    buffer = 50
    start_frame = int(start_frame) - int(buffer)
    end_frame = int(start_frame) + int(buffer) #  We don't really need to go past.  ... but just incase

    print(start_frame)
    print(end_frame)

    print('Ripping Video')
    rip = RipVideo(video, int(start_frame), int(end_frame))
    rip.render()

    print('Running segmentation')
    seg = Motion(video, int(start_frame), int(end_frame))
    seg.render()

    print('isolating objects')
    iso = IsolateObjects(video=video)
    iso.render()

    ann = Draw(video)
    ann.draw_annotations()

@manager.command
def run_find_fish(video, event_measure_file=None):
    """
    Parse the event measure file and locate the fish for all annotations in the file.

    :param video:
    :param event_measure_file:
    :return:
    """



@manager.command
def run_delete_cache():
    import os
    import glob

    rgb_files = glob.glob('mov/rgb/*')
    bak_files = glob.glob('mov/bak/1/*')
    merge_files = glob.glob('mov/merge/*')
    for r, b, m in zip(rgb_files, bak_files, merge_files):
        os.remove(r)
        os.remove(b)
        os.remove(m)



@manager.command
def run_render_video(directory):
    """Render a video from directory of *.jpg"""

    ren = RenderVideo(directory)
    ren.render()


@manager.command
def clear_logs():
    """Delete the log files"""
    os.remove('logs/application.log')


@manager.command
def generate_yolo_labels(movie):
    """Given  a directory of images found in a video"""

    label_dir = config.get('MODEL', 'label_directory')
    image_dir = config.get('MODEL', 'load_directory')

    yolo = YoloTools()
    yolo.streaming_template_match(image_dir, label_dir, movie, draw_box=False, normalised_coords=True)


@manager.command
def directory_template_match():
    """Given  a directory of images found in a direcotry of templates"""

    template_dir = config.get('MODEL', 'template_directory')
    image_dir = config.get('MODEL', 'load_directory')

    import os

    subdir = [f.path for f in os.scandir(image_dir) if f.is_dir()]

    print(subdir)

    yolo = YoloTools()

    for d in subdir:
        print(d)
        yolo.directory_template_match(d, template_dir, draw_box=True, normalised_coords=True)


if __name__ == '__main__':
    manager.main()
