import os

def check_if_checkpoint_exists(path):
    if os.path.isdir(path):
        return any([file for file in os.listdir(path) if "checkpoint" in file])
    return False