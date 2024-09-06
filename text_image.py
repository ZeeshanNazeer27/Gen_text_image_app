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
            
            # Debugging information
            st.write(f"Type of result: {type(result)}")
            st.write(f"Type of result.images: {type(result.images)}")
            st.write(f"Type of result.images[0]: {type(result.images[0])}")

            # Display raw result for inspection
            if isinstance(result.images, list):
                st.write(f"Length of result.images: {len(result.images)}")
                st.write("First few elements of result.images:", result.images[:1])
            else:
                st.write("result.images is not a list")

            # Extract the image
            image = result.images[0]
            
            # Convert image to a format Streamlit can display
            if isinstance(image, torch.Tensor):
                st.write("Result is a torch.Tensor")
                image = image.cpu().numpy()
                if image.ndim == 4:  # Handle batch dimension if present
                    image = image[0]
                if image.ndim == 3:
                    image = np.transpose(image, (1, 2, 0))  # Convert from CHW to HWC format
                image = Image.fromarray(image.astype(np.uint8))
            elif isinstance(image, np.ndarray):
                st.write("Result is a numpy.ndarray")
                if image.ndim == 4:  # Handle batch dimension if present
                    image = image[0]
                if image.ndim == 3:
                    image = np.transpose(image, (1, 2, 0))  # Convert from CHW to HWC format
                image = Image.fromarray(image.astype(np.uint8))
            elif isinstance(image, Image.Image):
                st.write("Result is a PIL.Image")
                # No conversion needed
                pass
            else:
                raise TypeError("Output is in an unsupported format")

            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
