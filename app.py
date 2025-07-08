import streamlit as st
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="Image Compression with PCA + KMeans", layout="centered")

st.title("🖼️ Image Compression using PCA + KMeans")
st.write("Upload an image and select number of colors to compress it.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Color selection
k = st.slider("Number of colors (KMeans clusters)", min_value=2, max_value=32, value=8)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.subheader("Original Image")
    st.image(img, caption="Original Image", use_column_width=True)

    # Preprocess
    pxl = img_np.reshape(-1, 3)
    pxl_scaled = pxl / 255.0

    # PCA
    pca = PCA(n_components=2)
    pxl_pca = pca.fit_transform(pxl_scaled)

    # KMeans
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(pxl_pca)
    centers_rgb = pca.inverse_transform(model.cluster_centers_)
    centers_rgb = (centers_rgb * 255).astype('uint8')
    labels = model.labels_
    compressed_pxl = centers_rgb[labels]
    compressed_img = compressed_pxl.reshape(img_np.shape)

    # Display compressed image
    st.subheader(f"Compressed Image with {k} Colors")
    st.image(compressed_img, use_column_width=True)

    # Download button
    compressed_pil = Image.fromarray(compressed_img)
    buf = io.BytesIO()
    compressed_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Compressed Image", data=byte_im, file_name="compressed.png", mime="image/png")
