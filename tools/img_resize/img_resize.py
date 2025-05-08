#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/ale/Downloads/ale/"
save_path = "/home/ale/Downloads/ale/imgs/train/"
dirs = os.listdir( path )
w = 640
h = 360

def resize():
    total = len(dirs)
    i = 0
    for item in dirs:
        if os.path.isfile(path+item):
            progress = 100*i/total
            print(f"{progress:.0f}%")
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            filename = item.split('.')[0]
            imResize = im.resize((w,h), Image.ANTIALIAS)
            imResize.save(save_path + filename + e)
            i += 1

resize()