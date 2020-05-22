import logging

from utils import file_exists

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_model_sources = dict()


def model_src_type(tp):
    def decorator(cls):
        global _model_sources
        _model_sources[tp] = cls
        return cls

    return decorator


class ModelSource(object):
    """
    A base class for all model sources
    """

    def __init__(self):
        self.__type = None
        self.uri = None
        self.client = None
        self.client_fs = None

    def load(self):
        from utils import load, loads
        with self.get_stream('rb') as stream:
            # TODO: Don`t load the whole data.
            unpickled = loads(stream.read())
            return unpickled

    def get_stream(self, flags='rw'):
        raise Exception('Stub')

    def delete(self):
        raise Exception('Stub')

    def set_user_client(self, client):
        from helpers import ClientFs
        self.client = client
        self.client_fs = ClientFs(client)

    def exists(self):
        raise Exception('Stub')

    def copy_to(self, destination):
        """
        Copy this model to the destination
        :param destination: ModelSource
        :return:
        """
        if isinstance(destination, str):
            destination = get_model_source('fs', destination)
        with self.get_stream('rb') as src_stream:
            with destination.get_stream('wb') as dst_stream:
                buff_size = 1024 * 10
                while (True):
                    read_bytes = src_stream.read(buff_size)
                    if read_bytes is not None and len(read_bytes) == 0:
                        break
                    if read_bytes is not None:
                        dst_stream.write(read_bytes)
        return True


@model_src_type("s3")
class S3ModelSource(ModelSource):

    def __init__(self, uri=None, anon=False):
        import s3fs
        super().__init__()
        self.__type = "s3"
        self.uri = uri
        self.s3 = s3fs.S3FileSystem(anon=anon)

    def get_stream(self, flags='rb'):
        import os
        fs_object = self.s3.open(self.uri, flags)
        return fs_object

    def exists(self):
        return self.s3.exists(self.uri)

    def delete(self):
        import os
        if self.s3.exists(self.uri):
            self.s3.rm(self.uri)
        else:
            return False
        return True


@model_src_type("fs")
class FSModelSource(ModelSource):

    def __init__(self, file_uri=None, abs=False):
        import os
        super().__init__()
        if not abs:
            file_uri = os.path.abspath(file_uri)
        self.__type = "fs"
        self.uri = file_uri
        #

    def get_stream(self, flags='rw'):
        f = open(self.uri, flags, buffering=0)
        return f

    def exists(self):
        return file_exists(self.uri)

    def delete(self):
        import os
        if os.path.exists(self.uri):
            os.remove(self.uri)
        else:
            return False
        return True


def get_local_model_source(model_path) -> ModelSource:
    """
    Gets the source of a model
    :param model_path:
    :return:
    """
    src = get_model_source('fs', model_path)
    return src


def get_model_source(tp, *args, **kwargs) -> ModelSource:
    """

    :param tp: The type of the source
    :param args:
    :param kwargs:
    :return:
    """
    global _model_sources
    cls = _model_sources.get(tp, None)
    if cls is not None:
        instance = cls(*args, **kwargs)
        # instance.__type = tp
        return instance
    else:
        return None


def import_model(msg):
    """
    Import a model from a source type storage, to the local fs storage
    :param msg:
    :return:
    """
    params = msg['params']
    src_type = params['type']
    src_addr = params['uri']
    for_client = params['client']
    source = get_model_source(src_type, src_addr)
    local_storage = get_model_source('fs')
    local_storage.set_user_client(for_client)
    output = source.copy_to(local_storage)
    return output
