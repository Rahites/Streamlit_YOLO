# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from check_rule import check_location

# Setting page layout
st.set_page_config(
    page_title="Kuaicv Helmet Detecting",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Helmet Detection using AI Detection Model")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path_person = Path(settings.DETECTION_MODEL_PERSON)
    model_path_helmet = Path(settings.DETECTION_MODEL_HELMET)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model_person, model_helmet = helper.load_model(model_path_person, model_path_helmet)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    class_count = [0, 0, 0, 0, 0, 0]

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):                
                res_person = model_person.predict(uploaded_image, conf=confidence, iou=0.35, classes=[0])
                res_helmet = model_helmet.predict(uploaded_image, conf=confidence, iou=0.8)

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

                res_plotted = res_helmet[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                # try:
                #     with st.expander("Detection Results"):
                #         for box in boxes_helmet:
                #             st.write(box.data)
                # except Exception as ex:
                #     # st.write(ex)
                #     st.write("No Helmet is in the Image!")
        
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
    
    st_text = [st_text1, st_text2]
    st_md_list = [st_md11, st_md12, st_md21, st_md22]

    st_text[0].markdown("<h2 style='color: black; font-size: 35px;'>Person : {}</h2>".format(class_count[4]), unsafe_allow_html=True)
    st_text[1].markdown("<h2 style='color: black; font-size: 35px;'>Helmeted Person : {}</h2>".format(class_count[5]), unsafe_allow_html=True)
    st_md_list[0].markdown("<h3 style='color: blue; font-size: 30px;'>blue : {}</h3>".format(class_count[0]), unsafe_allow_html=True)
    st_md_list[1].markdown("<h3 style='color: red; font-size: 30px;'>red : {}</h3>".format(class_count[1]), unsafe_allow_html=True)
    st_md_list[2].markdown("<h3 style='color: lightgray; font-size: 30px;'>white : {}</h3>".format(class_count[2]), unsafe_allow_html=True)
    st_md_list[3].markdown("<h3 style='color: #FFD700; font-size: 30px;'>yellow : {}</h3>".format(class_count[3]), unsafe_allow_html=True)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model_person, model_helmet)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model_person, model_helmet)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
