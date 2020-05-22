import envir
import os
import utils
import pandas as pd


class ClientFs(object):
    def __init__(self, client):
        self.client = client
        self.root = envir.get_assets_dir(client)

    def set_client(self, client):
        self.client = client
        self.root = envir.get_assets_dir(client)

    def save(self, path, content):
        if isinstance(path, list):
            path = os.path.join(*path)
        file_path = os.path.join(self.root, path)
        dir_path = os.path.dirname(file_path)
        utils.mkdir_p(dir_path)
        file_handle = open(file_path, 'w')
        file_handle.write(content)
        file_handle.close()

    def save_pickle(self, path, content, is_abs=False):
        from utils import save
        if not is_abs:
            if isinstance(path, list):
                path = os.path.join(*path)
            file_path = os.path.join(self.root, path)
            dir_path = os.path.dirname(file_path)
            utils.mkdir_p(dir_path)
        else:
            file_path = path

        save(content, file_path)
        return file_path

    def dumpdf(self, path, df):
        if isinstance(path, list):
            path = os.path.join(*path)
        file_path = os.path.join(self.root, path)
        dir_path = os.path.dirname(file_path)
        utils.mkdir_p(dir_path)
        df.to_csv(file_path)
        return file_path

    def load_pickle(self, path):
        from utils import load
        if isinstance(path, list):
            path = os.path.join(*path)
        file_path = os.path.join(self.root, path)
        loaded = load(file_path)
        return loaded

    def get(self, path):
        """
        Gets an absolute path related to a client.
        Makes sure that the directory of the path exists
        :param path:
        :return:
        """
        if isinstance(path, list):
            path_end = os.path.join(*path)
            file_path = os.path.join(self.root, path_end)
            dir_path = os.path.dirname(file_path)
            utils.mkdir_p(dir_path)
        else:
            file_path = os.path.join(self.root, path)
        return file_path
