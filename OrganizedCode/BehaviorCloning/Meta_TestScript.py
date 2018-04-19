#!/usr/bin/env python
from headers import *

# for i in range(0,260,10):
for i in range(0,260):
	print("###############################################")
	print("STARTING WITH:",i)
	# FOR FULL PARSING:
	# command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --train 0 --gpu 0,1 --depth_terminate 1 --model saved_models/model_epoch{0}.ckpt --suffix model_dt4_{0}".format(i)
	# command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --train 0 --gpu 2,3 --depth_terminate 1 --mix 0 --model saved_models/model_epoch{0}.ckpt --suffix model_mix_{0}".format(i)
	command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --indices ../../../ImageSets/TrainIndices.npy --train 0 --gpu 3,1 --depth_terminate 1 --model saved_models/model_epoch{0}.ckpt --suffix model_mix_{0}".format(i)
	subprocess.call(command.split(),shell=False)
