#!/usr/bin/env python
from headers import *

for i in range(0,510,5):
# for i in range(8,60):
	print("###############################################")
	print("STARTING WITH:",i)
	# FOR FULL PARSING:
	# command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --train 0 --gpu 0,1 --depth_terminate 1 --model saved_models/model_epoch{0}.ckpt --suffix model_dt4_{0}".format(i)
	# command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350BINLABS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --train 0 --gpu 2,3 --depth_terminate 1 --mix 0 --model saved_models/model_epoch{0}.ckpt --suffix model_mix_{0}".format(i)

    # TEST AND TRAIN SET
	command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350IMS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --indices ../../../ImageSets/TrainIndices.npy --train 0 --gpu 4,3 --model saved_models/model_epoch{0}.ckpt --suffix model_trainval_{0}".format(i)
	subprocess.call(command.split(),shell=False)
	command = "python ../Meta_RLClass.py --images ../../../ImageSets/NEW350IMS.npy --labels ../../../ImageSets/NEW350BINLABS.npy --indices ../../../ImageSets/TestIndices.npy --train 0 --gpu 4,3 --model saved_models/model_epoch{0}.ckpt --suffix model_test_{0}".format(i)
	subprocess.call(command.split(),shell=False)

	# # FOR RANDOMSQUARES
	# command = "python ../Meta_RLClass.py --images ../../../ImageSets/Images_RandomSquares.npy --labels ../../../ImageSets/Images_RandomSquares.npy --train 0 --gpu 3,2 --model saved_models/model_epoch{0}.ckpt --suffix model_trainval_{0}".format(i)
	# subprocess.call(command.split(),shell=False)
