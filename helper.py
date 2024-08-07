from ultralytics import YOLO
import streamlit as st
import cv2
import time
import yt_dlp
import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, st_text, st_md_list, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    class_count = [0, 0, 0, 0, 0]
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    # TODO
    # print('class : ', res[0].boxes.cls)
    try:
        with st.expander("Detection Results"):
            boxes = res[0].boxes
            for box in boxes:
                box_class = box.cls
                # 0: blue, 1: red, 2: white, 3: yellow, 4: person
                if 0 in box_class:
                    class_count[0] += 1
                if 1 in box_class:
                    class_count[1] += 1
                if 2 in box_class:
                    class_count[2] += 1
                if 3 in box_class:
                    class_count[3] += 1
                if 4 in box_class: 
                    class_count[4] += 1
                # st_text.markdown(f"### Person : {class_count[4]}")
                # st_md_list[0].markdown(f"### blue : {class_count[0]}")
                # st_md_list[1].markdown(f"### red : {class_count[1]}")
                # st_md_list[2].markdown(f"### white : {class_count[2]}")
                # st_md_list[3].markdown(f"### yellow : {class_count[3]}")
                st_text.markdown("<h2 style='color: black; font-size: 35px;'>Person : {}</h2>".format(class_count[4]), unsafe_allow_html=True)
                st_md_list[0].markdown("<h3 style='color: blue; font-size: 30px;'>blue : {}</h3>".format(class_count[0]), unsafe_allow_html=True)
                st_md_list[1].markdown("<h3 style='color: red; font-size: 30px;'>red : {}</h3>".format(class_count[1]), unsafe_allow_html=True)
                st_md_list[2].markdown("<h3 style='color: lightgray; font-size: 30px;'>white : {}</h3>".format(class_count[2]), unsafe_allow_html=True)
                st_md_list[3].markdown("<h3 style='color: #FFD700; font-size: 30px;'>yellow : {}</h3>".format(class_count[3]), unsafe_allow_html=True)
    except Exception as ex:
        st.write("Nothing Detected")



def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)
            st_frame = st.empty()
            st_text = st.empty()

            col1, col2 = st.columns(2)
            with col1:
                st_md11 = st.empty()
                st_md12 = st.empty()
            with col2:
                st_md21 = st.empty()
                st_md22 = st.empty()

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")

            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             [st_md11, st_md12, st_md21, st_md22],
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            st_text = st.empty()

            col1, col2 = st.columns(2)
            with col1:
                st_md11 = st.empty()
                st_md12 = st.empty()
            with col2:
                st_md21 = st.empty()
                st_md22 = st.empty()
            
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             [st_md11, st_md12, st_md21, st_md22],
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            st_text = st.empty()

            col1, col2 = st.columns(2)
            with col1:
                st_md11 = st.empty()
                st_md12 = st.empty()
            with col2:
                st_md21 = st.empty()
                st_md22 = st.empty()
            
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             [st_md11, st_md12, st_md21, st_md22],
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # 프레임 간 이상적인 시간 간격 계산 (초 단위)
            frame_interval = 1 / fps

            st_frame = st.empty()
            st_text = st.empty()

            col1, col2 = st.columns(2)
            with col1:
                st_md11 = st.empty()
                st_md12 = st.empty()
            with col2:
                st_md21 = st.empty()
                st_md22 = st.empty()
            
            start_time = time.time()
            frame_count = 0
            
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             [st_md11, st_md12, st_md21, st_md22],
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
                
                # 영상 시간에 맞게 Delay
                frame_count += 1
                elapsed_time = time.time() - start_time
                expected_time = frame_count * frame_interval
                print('delay : ', expected_time - elapsed_time)
                if elapsed_time < expected_time:
                    time.sleep(expected_time - elapsed_time)

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
