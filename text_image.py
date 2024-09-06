import streamlit as st
import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
@st.cache_resource
def load_model():
    # Load the stable diffusion pipeline
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
    pipe = pipeline.to("cuda")
    return pipe

# Streamlit app layout
st.title("Text-to-Image Generator")

# User input for the text prompt
prompt = st.text_input("Enter your prompt for the image generation:", "a cricket stadium with match going on in it")

# Button to generate the image
if st.button("Generate Image"):
    # Load the model
    pipe = load_model()

    # Display a message while generating the image
    with st.spinner('Generating image...'):
        image = pipe(prompt).images[0]

    # Save and display the generated image
    image.save("generated_image.png")
    st.image(image, caption="Generated Image", use_column_width=True)

    # Provide download option
    with open("generated_image.png", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="generated_image.png",
            mime="image/png"
        )
