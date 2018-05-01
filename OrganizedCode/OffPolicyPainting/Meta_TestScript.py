#!/usr/bin/env python
from headers import *

# for i in range(0,260,10):
for i in range(1,100):
# for i in range(10,260,10):
	print("###############################################")
	print("STARTING WITH:",i)
	# FOR FULL PARSING:
	command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --paint 1 --train 0 --gpu 0,1 --plot 0 --model saved_models/model_epoch{0}.ckpt --suffix model_pw50_{0}".format(i)
	subprocess.call(command.split(),shell=False)
