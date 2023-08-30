"""
This is the main streamlit app responsible for running all the code related to learning a yoga pose.
"""

import streamlit as st

import av
import cv2
import numpy as np
import math
import json
import time
import os

from pose_estimator import PoseEstimator
from widgets import *
from _callbacks import *
from helpers import *

from typing import List, Literal

poses, pose_dict = read_poses_json()


display_reference_image(poses)
pose = camera_preview(poses)
timed_session()  
#