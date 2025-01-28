import cv2
import numpy as np
from PIL import Image, ImageFile
import random
import streamlit as st

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to load an image
def load_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to save an image (convert OpenCV to Pillow)
def save_image(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Uniform bubble distortion with random selection
def uniform_bubble_distortion(image, num_bubbles=16, selected_fraction=0.5, min_bubble_size=50, max_bubble_size=150, min_strength=0.1, max_strength=0.5):
    h, w = image.shape[:2]
    distorted_image = image.copy()

    # Divide the image into a uniform grid based on the number of bubbles
    grid_rows = int(np.sqrt(num_bubbles))
    grid_cols = int(np.ceil(num_bubbles / grid_rows))

    # Calculate grid cell size
    cell_height = h // grid_rows
    cell_width = w // grid_cols

    # Create a list of all possible bubble regions (center points)
    bubble_centers = [
        (j * cell_width + cell_width // 2, i * cell_height + cell_height // 2)
        for i in range(grid_rows)
        for j in range(grid_cols)
    ]

    # Randomly select a fraction of bubbles to apply the effect
    num_selected_bubbles = int(len(bubble_centers) * selected_fraction)
    selected_bubbles = random.sample(bubble_centers, num_selected_bubbles)

    for x_center, y_center in selected_bubbles:
        # Randomize bubble size and strength
        bubble_size = random.randint(min_bubble_size, max_bubble_size)
        strength = random.uniform(min_strength, max_strength)

        # Ensure bubble fits within the cell
        bubble_size = min(bubble_size, cell_width // 2, cell_height // 2)

        # Apply the bubble effect
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        factor = 1 + strength * np.exp(-distance**2 / (2 * bubble_size**2))

        x_new = (x * factor).astype(np.float32)
        y_new = (y * factor).astype(np.float32)

        distorted_image = cv2.remap(distorted_image, x_new, y_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return distorted_image

# Streamlit app
def main():
    st.title("🦀Yengec Web App 🦀")
    st.write("Upload an image and apply a fun bubble distortion effect!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = load_image(uploaded_file)

        if image is not None:
            st.image(save_image(image), caption="Original Image", use_column_width=True)

            # Parameters
            st.sidebar.header("Distortion Parameters")
            num_bubbles = st.sidebar.slider("Number of Bubbles", 4, 64, 16)
            selected_fraction = st.sidebar.slider("Fraction of Bubbles to Apply Effect", 0.1, 1.0, 0.5)
            min_bubble_size = st.sidebar.slider("Min Bubble Size", 10, 200, 50)
            max_bubble_size = st.sidebar.slider("Max Bubble Size", 50, 300, 150)
            min_strength = st.sidebar.slider("Min Distortion Strength", 0.0, 1.0, 0.1)
            max_strength = st.sidebar.slider("Max Distortion Strength", 0.1, 1.0, 0.3)

            # Apply distortion button
            if st.button("Apply Distortion"):
                distorted_image = uniform_bubble_distortion(
                    image,
                    num_bubbles=num_bubbles,
                    selected_fraction=selected_fraction,
                    min_bubble_size=min_bubble_size,
                    max_bubble_size=max_bubble_size,
                    min_strength=min_strength,
                    max_strength=max_strength
                )

                st.image(save_image(distorted_image), caption="Distorted Image", use_column_width=True)

                # Download button
                result_image = save_image(distorted_image)
                st.download_button(
                    label="Download Distorted Image",
                    data=result_image.tobytes(),
                    file_name="distorted_image.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
