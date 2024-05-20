import streamlit as st
import cv2
import base64
import os
from openai import OpenAI

# Initialize OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("Please set the OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# Streamlit app title
st.title("Posture Analysis from Video")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the video file
    video = cv2.VideoCapture("uploaded_video.mp4")
    base64Frames = []

    # Read video frames and convert to base64
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    st.write(f"{len(base64Frames)} frames read.")

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "Please analyze the posture of the person sitting for the interview in these video frames. "
                "Assess their current posture and provide advice on how they can improve it in four specific points. "
                "Additionally, rate their posture out of 10. Consider factors such as alignment, position of the back and shoulders, "
                "head placement, and overall body language. Your analysis should help the individual appear more confident, engaged, "
                "and professional during the interview. Include any specific observations that could be beneficial for their improvement.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
            ],
        },
    ]

    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    # Call the OpenAI API and get the result
    result = client.chat.completions.create(**params)
    analysis = result.choices[0].message.content

    # Display the result in Streamlit
    st.subheader("Posture Analysis Result")
    st.write(analysis)
else:
    st.write("Please upload a video file to analyze.")
