#!/usr/bin/env python
import numpy as npy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

import random
import sys
import copy
import os
import shutil
import subprocess
import glob
import argparse

import tensorflow as tf
import cv2

import random
import cv2
import h5py
from IPython import embed
from scipy.stats import multivariate_normal
