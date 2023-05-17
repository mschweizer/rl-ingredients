import os
import shutil
from logging import Logger

from sacred.observers import FileStorageObserver
from sacred.utils import PathType


class CustomFileStorageObserver(FileStorageObserver):
    def __init__(self, basedir: PathType):
        self.dir = None
        super().__init__(basedir)

    def get_log_path_with_new_name(self, new_name: str, logger: Logger):
        old_dir = self.dir
        new_dir = os.path.join(self.basedir, str(new_name))
        assert not os.path.exists(new_dir), f"Log directory '{new_dir}' already exists."
        self.dir = shutil.move(self.dir, new_dir)
        logger.info(f"Renamed log directory from '{old_dir}' to '{new_dir}'")
        return self.dir
