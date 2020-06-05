import pickle
import cv2
import os
import time

from argparse import ArgumentParser

def get_label_color(label):
    color_wheel = {
        "abalistes_stellatus": (0xFF, 0xFF, 0xFF),
        "acanthurus_triostegus": (0x77, 0x66, 0xCC),
        "epinephelus_areolatus": (0x77, 0xCC, 0xDD),
        "epinephelus_multinotatus": (0x33, 0x77, 0x11),
        "fish": (0x00, 0x00, 0xFF),
        "lethrinus_atkinsoni": (0x99, 0x44, 0xAA),
        "lethrinus_punctulatus": (0x00, 0xFF, 0x00),
        "lutjanus_bohar": (0x33, 0x99, 0x99),
        "lutjanus_sebae": (0x55, 0x22, 0x88),
        "lutjanus_vitta": (0x00, 0x11, 0x66),
        "pentapodus_porosus": (0x00, 0xFF, 0xFF),
        "pomacentrus_coelestis": (0x88, 0x88, 0x88),
        "thalassoma_lunare": (0xFF, 0xFF, 0xFF) 
    }

    return color_wheel[label]

def main(args):
    with open(args.detections_path, 'rb') as detection_file:
        frame_numbers, frame_detections = pickle.load(detection_file)

    reader = cv2.VideoCapture(args.video_path)
    filename, ext = os.path.splitext(os.path.basename(args.video_path))

    # Determine the output path and ensure all intermediate directories exist
    output_path = os.path.abspath(args.output_path or os.path.join(os.path.dirname(args.video_path), filename + '.detections' + ext))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not reader.isOpened():
        print('error: unable to open "%s"; please ensure it exists' % args.video_path)
        return 1

    frame_size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_index = 0
    frame_number = 0
    detections = []

    # create a video writer to write the new frames to
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), reader.get(cv2.CAP_PROP_FPS), frame_size)
    
    # re-usable colors
    white = (255, 255, 255)
    black = (0, 0, 0)

    start_time = time.time()

    while reader.isOpened():
        res, frame = reader.read()

        if not res:
            break

        if frame_index < len(frame_numbers) and frame_numbers[frame_index] == frame_number:
            detections = frame_detections[frame_index]
            frame_index += 1

        if (frame_number % 1000) == 0:
            print('info: processed up to frame %d' % frame_number)

        totals = dict()

        for label, prob, x1, y1, x2, y2 in detections:
            color = get_label_color(label)
            totals[label] = 1 if not label in totals else totals[label] + 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, '%s: %.3f' % (label, prob), (x1, y1 - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, white, 1)

        # draw totals
        text_pos = (15, frame_size[1] - 30)

        cv2.putText(frame, 'total: %d' % len(detections), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, 4)
        cv2.putText(frame, 'total: %d' % len(detections), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        for item in sorted(totals.items(), reverse=True, key=lambda x: x[1]):
            text_pos = (text_pos[0], text_pos[1] - 30)
            cv2.putText(frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, 4)
            cv2.putText(frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        writer.write(frame)
        frame_number +=1

    duration = time.time() - start_time
    print('info: completed in %.2fs' % duration)
    print('info: average fps %.2f' % (frame_number / duration))

    writer.release()
    reader.release()

if __name__ == '__main__':
    parser = ArgumentParser('render_detections')
    parser.add_argument('-o', '--output-path', help='Path to output the video to (defaults to the same directory as the input video)', default=None)
    parser.add_argument('detections_path', help='Path to pre-generated Pickle detections file')
    parser.add_argument('video_path', help='Path to video file to run the detections on')

    args = parser.parse_args()
    exit(main(args) or 0)