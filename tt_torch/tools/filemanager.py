# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import shutil
import os

# This should be in its own file to prevent a rare deadlock condition due to ttmlir input with tracy


class FileManager:
    @staticmethod
    def create_file(file_path):
        try:
            if not FileManager.check_directory_exists(os.path.dirname(file_path)):
                FileManager.create_directory(os.path.dirname(file_path))
            with open(file_path, "w") as file:
                file.write("")
        except OSError as e:
            raise OSError(f"error creating file: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def create_directory(directory_path, exist_ok=False):
        try:
            os.makedirs(directory_path, exist_ok=exist_ok)
        except FileExistsError as e:
            raise FileExistsError(f"directory '{directory_path}' already exists")
        except OSError as e:
            raise OSError(f"error creating directory: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_file(file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"directory '{file_path}' not found - cannot remove")
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove file '{file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_directory(directory_path):

        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            print(f"directory '{directory_path}' not found - cannot remove")
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove directory '{directory_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def copy_file(dest_file_path, src_file_path):
        try:
            shutil.copy2(src_file_path, dest_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"the source file does not exist: '{src_file_path}'"
            )
        except PermissionError as e:
            raise PermissionError(
                f"permission denied: '{src_file_path}' or '{dest_file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def check_file_exists(file_path):
        exists = False
        try:
            if os.path.exists(file_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    @staticmethod
    def check_directory_exists(directory_path):
        exists = False
        try:
            if os.path.isdir(directory_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists
