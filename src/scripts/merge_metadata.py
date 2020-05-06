import os
import json

from argparse import ArgumentParser

def main(args):
    if not os.path.isdir(args.directory):
        print('error: unable to find directory "%s"' % args.directory)
        return 1

    master = { '_via_img_metadata': {} }
    root = master['_via_img_metadata']

    with os.scandir(args.directory) as scanner:
        for entry in scanner:
            if entry.is_file() and entry.name.endswith('.json'):
                with open(entry.path, 'r') as f:
                    try:
                        content = json.load(f)
                        
                        if '_via_img_metadata' not in content:
                            raise Exception('Missing required key')
                        
                        for key in content['_via_img_metadata']:
                            source = content['_via_img_metadata'][key]

                            # use 'filename' as the key to avoid submitting the same file twice
                            if 'filename' in source and source['filename'] is not None:
                                root[source['filename']] = source
                    except Exception:
                        print('warning: invalid JSON found in "%s"' % entry.name)
                        
    with open('./master.json', 'w') as f:
        json.dump(master, f)

if __name__ == '__main__':
    parser = ArgumentParser('merge_metadata', 'Utility to merge frame metadata files together')
    parser.add_argument('directory', help='Directory containing the metadata files to merge')

    args = parser.parse_args()
    exit(main(args) or 0)