#!/usr/bin/env python
import numpy as npy
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
from shapely.geometry import box
from shapely.geometry import point
from shapely.affinity import rotate