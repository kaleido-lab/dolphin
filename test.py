from modules.sadtalker import Sadtalker
import os
import sys

print(os.path.split(sys.argv[0])[0])
ad = Sadtalker()
ad.inference('./image/bus_chinese.wav,./image/art_0.png')

