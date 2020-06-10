""" Useful utility functions that don't belong anywhere else """

from errno import ENOENT

import imagesize
import re

class FriendlyError(Exception):
    """ 
    An error with a user-friendly message, useful for denoting errors which may be
    acceptable to present to the end-user.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

def _normalize_header(header):
    return re.sub(r'[^\w]+', '_', header.lower())

def get_image_size(path):
    try:
        return imagesize.get(path)
    except (OSError, IOError) as err:
        if getattr(err, 'errno', 0) != ENOENT:
            raise
        return None

def read_normalized_csv_headers(csv_reader):
    """
    Read the next row from a CSV reader, interpret it as table headers and return a dict mapping them to their indices
    """
    headers = next(csv_reader)
    return {_normalize_header(headers[i]): i for i in range(len(headers))}