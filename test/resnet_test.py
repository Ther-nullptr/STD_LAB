import numpy as np
import sys

sys.path.append('..')
from Vfeat_extracter import *

if __name__ == '__main__':
    path = '/mnt/c/Users/86181/Desktop/STD-project/test/input_mp4_file.mp4'
    output = extract_video(path)
