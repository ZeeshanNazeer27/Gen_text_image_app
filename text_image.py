import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np

# Load the stable diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
pipe = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit app
st.title("Text-to-Image Generator")

# Text prompt input
prompt = st.text_input("Enter a text prompt for image generation:", "a cricket stadium with match going on in it")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            # Generate the image
            result = pipe(prompt)
            image = result.images[0]
            
            # Ensure image is in PIL format
            if isinstance(image, torch.Tensor):
                # Convert torch.Tensor to PIL.Image
                image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
            elif isinstance(image, np.ndarray):
                # Convert numpy.ndarray to PIL.Image
                image = Image.fromarray(image.astype(np.uint8))
            elif not isinstance(image, Image.Image):
                raise TypeError("Output is in an unsupported format")
            
            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
