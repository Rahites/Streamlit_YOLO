from ultralytics import YOLO
from check_rule import check_location
import streamlit as st
import cv2
import time
import yt_dlp
import settings


def load_model(model_path_person, model_path_helmet):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model_path_person = YOLO(model_path_person)
    model_path_helmet = YOLO(model_path_helmet)
    return model_path_person, model_path_helmet


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model_person, model_helmet, st_frame, st_text, st_md_list, image, is_display_tracking=None, tracker=None):
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
    class_count = [0, 0, 0, 0, 0, 0]
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # is_display_tracking = False

    # # Display object tracking, if specified
    # if is_display_tracking:
    #     res = model.track(image, conf=conf, persist=True, tracker=tracker)
    # else:
    #     # Predict the objects in the image using the YOLOv8 model
    res_person = model_person.predict(image, conf=conf, iou=0.35, classes=[0])
    res_helmet = model_helmet.predict(image, conf=conf, iou=0.8)

    # # Plot the detected objects on the video frame
    res_person_plotted = res_person[0].plot()
    res_helmet_plotted = res_helmet[0].plot()

    plot_blended = cv2.addWeighted(res_person_plotted, 0.5, res_helmet_plotted, 0.5, 0)
    st_frame.image(plot_blended,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    # TODO
    # print('class : ', res[0].boxes.cls)
    try:
        with st.expander("Detection Results"):
            boxes_person = res_person[0].boxes
            boxes_helmet = res_helmet[0].boxes
            boxes_p_list, boxes_h_list = [], []
            for box_p in boxes_person:                    
                if 0 in box_p.cls: 
                    # 0: person
                    class_count[4] += 1
                boxes_p_list.append(box_p.xywh)
            for box_h in boxes_helmet:
                # 0: blue, 1: red, 2: white, 3: yellow
                if 0 in box_h.cls:
                    class_count[0] += 1
                if 1 in box_h.cls:
                    class_count[1] += 1
                if 2 in box_h.cls:
                    class_count[2] += 1
                if 3 in box_h.cls:
                    class_count[3] += 1    
                boxes_h_list.append(box_h.xywh)                                    
            helmeted_person = check_location(boxes_p_list, boxes_h_list)
            class_count[5] = helmeted_person
            st_text[0].markdown("<h2 style='color: black; font-size: 35px;'>Person : {}</h2>".format(class_count[4]), unsafe_allow_html=True)
            st_text[1].markdown("<h2 style='color: black; font-size: 35px;'>Helmeted Person : {}</h2>".format(class_count[5]), unsafe_allow_html=True)
            st_md_list[0].markdown("<h3 style='color: blue; font-size: 30px;'>blue : {}</h3>".format(class_count[0]), unsafe_allow_html=True)
            st_md_list[1].markdown("<h3 style='color: red; font-size: 30px;'>red : {}</h3>".format(class_count[1]), unsafe_allow_html=True)
            st_md_list[2].markdown("<h3 style='color: lightgray; font-size: 30px;'>white : {}</h3>".format(class_count[2]), unsafe_allow_html=True)
            st_md_list[3].markdown("<h3 style='color: #FFD700; font-size: 30px;'>yellow : {}</h3>".format(class_count[3]), unsafe_allow_html=True)
    except Exception as ex:
        print(ex)
        st_text[0].markdown("<h2 style='color: black; font-size: 35px;'>Person : {}</h2>".format(class_count[4]), unsafe_allow_html=True)
        st_text[1].markdown("<h2 style='color: black; font-size: 35px;'>Helmeted Person : {}</h2>".format(class_count[5]), unsafe_allow_html=True)
        st_md_list[0].markdown("<h3 style='color: blue; font-size: 30px;'>blue : {}</h3>".format(class_count[0]), unsafe_allow_html=True)
        st_md_list[1].markdown("<h3 style='color: red; font-size: 30px;'>red : {}</h3>".format(class_count[1]), unsafe_allow_html=True)
        st_md_list[2].markdown("<h3 style='color: lightgray; font-size: 30px;'>white : {}</h3>".format(class_count[2]), unsafe_allow_html=True)
        st_md_list[3].markdown("<h3 style='color: #FFD700; font-size: 30px;'>yellow : {}</h3>".format(class_count[3]), unsafe_allow_html=True)
        pass # st.write("Nothing Detected")



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


def play_webcam(conf, model_person, model_helmet):
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
            col1_t, col2_t = st.columns(2)
            with col1_t:
                st_text1 = st.empty()
            with col2_t:
                st_text2 = st.empty()

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
                                             model_person, 
                                             model_helmet,
                                             st_frame,
                                             [st_text1, st_text2],
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


def play_stored_video(conf, model_person, model_helmet):
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
            col1_t, col2_t = st.columns(2)
            with col1_t:
                st_text1 = st.empty()
            with col2_t:
                st_text2 = st.empty()

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
                                             model_person, 
                                             model_helmet,
                                             st_frame,
                                             [st_text1, st_text2],
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
