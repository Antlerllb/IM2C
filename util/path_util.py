import os


class PathUtil(object):
    @staticmethod
    def get_project_path():
        return os.path.dirname(os.path.dirname(__file__))

    @staticmethod
    def get_abspath(sub_path):
        return os.path.join(PathUtil.get_project_path(), sub_path)

    @staticmethod
    def get_output_path(sub_path):
        # problem1.png
        # .../output/subpath
        return os.path.join(PathUtil.get_project_path(), 'output',sub_path)