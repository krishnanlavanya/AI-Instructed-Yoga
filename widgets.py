import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode, create_mix_track, create_process_track
from streamlit_server_state import server_state, server_state_lock
from streamlit.components.v1 import html

from _callbacks import *

import base64


def timed_session():
    if st.button('Start Timed Session'):
        st.write('Your Timed Session Has Started')
        
        my_html = """
        <script>
        function startTimer(duration, display) {
            var timer = duration, minutes, seconds;
            setInterval(function () {
                minutes = parseInt(timer / 60, 10)
                seconds = parseInt(timer % 60, 10);

                minutes = minutes < 10 ? "0" + minutes : minutes;
                seconds = seconds < 10 ? "0" + seconds : seconds;

                display.textContent = minutes + ":" + seconds;

                if (--timer < 0) {
                    display.textContent = "";
                }
            }, 1000);
        }

        window.onload = function () {
            var fiveMinutes = 10,
                display = document.querySelector('#time');
            startTimer(fiveMinutes, display);
        };
        </script>

        <body>
        <div><span id="time"></span></div>
        </body>
        """
        html(my_html)

def camera_preview(poses):
    pose = None
    with server_state_lock["webrtc_contexts"]:
        if "webrtc_contexts" not in server_state:
            server_state["webrtc_contexts"] = []

        with server_state_lock["mix_track"]:
            if "mix_track" not in server_state:
                server_state["mix_track"] = create_mix_track(
                    kind="video", mixer_callback=mixer_callback, key="mix"
                )

        mix_track = server_state["mix_track"]
        self_ctx = webrtc_streamer(
                key="self",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                source_video_track=mix_track,
                sendback_audio=False,
            )

        self_process_track = None
        if self_ctx.input_video_track:
            self_process_track = create_process_track(
                input_track=self_ctx.input_video_track,
                processor_factory=VideoProcessor,
            )
            mix_track.add_input_track(self_process_track)
    
    return pose

def display_reference_image(poses):

    pose = st.radio(
        "Select the yoga pose to learn",
        poses,
        key="filter1-type",
    )
    with open('.current_pose.txt', 'w') as file:
        file.write(pose)
    try:
        with open(f'pose_images/{pose}.jpg', 'rb') as file:
            img = file.read()
            data_url = base64.b64encode(img).decode("utf-8")
        st.markdown("<div class=pose_image >", unsafe_allow_html=True,)
        st.markdown(f'<img class=pose_image src="data:image/gif;base64,{data_url}" height=200px width=150px style="margin-left:470px; margin-top:-200px">', unsafe_allow_html=True,)
        st.markdown("</div>", unsafe_allow_html=True,)
        with open('cache', 'w') as file:
            file.write(f'<img src="data:image/gif;base64,{data_url}" height=500px width=400px>')
        # st.image(f'pose_images/{pose}.jpg')
    except FileNotFoundError:
        pass