import processing.frames
import processing.file_handler


def get_frame_parser(tp, *args, **kwargs):
    """

    :param tp: The type of parser to use. Examples are : file, pandas, collection, s3.
    Each parser can have different arguments.
    :param args:
    :param kwargs:
    :return:
    """
    return processing.frames.get_frame_parser(tp, *args, **kwargs)
