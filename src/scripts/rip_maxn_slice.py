import os
import subprocess
from argparse import ArgumentParser
from math import floor


def main(args):
    if not os.path.isfile(args.video_path):
        print('error: unable to find file "%s"' % args.video_path)
        return 1
    filename, ext = os.path.splitext(os.path.basename(args.video_path))
    slice_path = os.path.join(os.path.dirname(args.video_path), os.path.join('.', '%s_%ds_slice%s' % (filename, args.length, ext)))
    frame_path = os.path.join(os.path.dirname(args.video_path), os.path.join('.', '%s_frame_%d.png' % (filename, args.count)))
    seconds = args.timestamp * 60
    start = max(0, seconds - (args.length / 2))
    ffmpeg_args = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning']
    print([*ffmpeg_args, '-ss', '%02.4f' % seconds, '-i', args.video_path, '-vframes', '1', frame_path])
    subprocess.run([*ffmpeg_args, '-ss', '%02.4f' % seconds, '-i', args.video_path, '-vframes', '1', frame_path])
    subprocess.run([*ffmpeg_args, '-ss', '%02.4f' % start, '-i', args.video_path, '-t', str(args.length), '-c', 'copy',
                    slice_path])


if __name__ == '__main__':
    parser = ArgumentParser('slice')
    parser.add_argument('-l', '--length', required=False, type=int, help='Length of the slice in seconds', default=60)
    parser.add_argument('video_path')
    parser.add_argument('timestamp', type=float, help='Timestamp of key frame expressed in minutes')
    parser.add_argument('count', type=int, help='Value for count in key frame')
    args = parser.parse_args()
    exit(main(args) or 0)
