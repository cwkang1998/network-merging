import os
# import shutil


def create_op_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        # shutil.rmtree(dir)
