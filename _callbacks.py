import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode, create_mix_track, create_process_track
from streamlit_server_state import server_state, server_state_lock

import av
import cv2
import numpy as np
import math

from pose_estimator import PoseEstimator

from typing import List, Literal

def mixer_callback(frames: List[av.VideoFrame]) -> av.VideoFrame:
    buf_w = 640
    buf_h = 480
    buffer = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

    n_inputs = len(frames)

    n_cols = math.ceil(math.sqrt(n_inputs))
    n_rows = math.ceil(n_inputs / n_cols)
    grid_w = buf_w // n_cols
    grid_h = buf_h // n_rows

    for i in range(n_inputs):
        frame = frames[i]
        if frame is None:
            continue

        grid_x = (i % n_cols) * grid_w
        grid_y = (i // n_cols) * grid_h

        img = frame.to_ndarray(format="bgr24")
        src_h, src_w = img.shape[0:2]

        aspect_ratio = src_w / src_h

        window_w = min(grid_w, int(grid_h * aspect_ratio))
        window_h = min(grid_h, int(window_w / aspect_ratio))

        window_offset_x = (grid_w - window_w) // 2
        window_offset_y = (grid_h - window_h) // 2

        window_x0 = grid_x + window_offset_x
        window_y0 = grid_y + window_offset_y
        window_x1 = window_x0 + window_w
        window_y1 = window_y0 + window_h

        buffer[window_y0:window_y1, window_x0:window_x1, :] = cv2.resize(
            img, (window_w, window_h)
        )

    new_frame = av.VideoFrame.from_ndarray(buffer, format="bgr24")

    return new_frame
    
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.style = 'color'    
        self.type = None
        self.pe = PoseEstimator(window_size=8, smoothing_function='savgol')

    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        print("Image", type(img))
        pose_coords = self.pe.get_pose_coords(img)
        print("counter", self.type)
        if pose_coords:
            angles = self.pe.get_angles(pose_coords)
            angle_colours = self.pe.get_angle_colour(angles)
            annotated_img = self.pe.get_annotated_image(img, pose_coords, angle_colours)
            st.subheader(f'{angles}')
        else:
            annotated_img = cv2.flip(img, 1)
        return av.VideoFrame.from_ndarray(annotated_img, format='bgr24')