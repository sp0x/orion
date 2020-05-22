import os
import utils
import sys

dir_assets = "assets"


def get_base_path():
    return "/experiments"


def get_experiments_dir(client):
    client_path = os.path.join(get_base_path(), client)
    utils.mkdir_p(client_path)
    return client_path


def get_assets_dir(client):
    client_path = os.path.join(get_base_path(), client, dir_assets)
    utils.mkdir_p(client_path)
    return client_path


class add_path:
    """

    """
    def __init__(self, path):
        """
        The path to use for context
        :param path:
        """
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
