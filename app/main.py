import os, glob
import streamlit as st
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Image Search", page_icon="🔎", layout="wide")
EMB_MODEL = "clip-ViT-B-32"
MODEL_PATH = os.path.join("model",EMB_MODEL)
INDEX_PATH = "index.faiss"
META_PATH = "meta.npy"


@st.cache_resource(show_spinner="Loading ...")
def load_model():
    if os.path.exists(MODEL_PATH):   
        model = SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer(EMB_MODEL) 
        model.save(MODEL_PATH)                 
    return model

model = load_model()

def embed_images(image_paths):
    imgs = []
    for p in image_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            imgs.append(None)
    return model.encode(imgs, convert_to_numpy=True, batch_size=32, show_progress_bar=True)

def embed_text(text):
    return model.encode([text], convert_to_numpy=True)[0]

def build_index(gallery_dir="gallery"):
    image_paths = sorted(glob.glob(os.path.join(gallery_dir, "**", "*.*"), recursive=True))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if not image_paths:
        st.warning(" No images found. Mount or place some under `./gallery`")
        return None, None, None
    with st.spinner(f"Embedding {len(image_paths)} images..."):
        embs = embed_images(image_paths)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embs)
    index.add(embs.astype(np.float32))
    np.save(META_PATH, np.array(image_paths, dtype=object))
    faiss.write_index(index, INDEX_PATH)
    return index, embs, image_paths

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        paths = np.load(META_PATH, allow_pickle=True).tolist()
        return index, paths
    return None, None

def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/751/751463.png", width=60)
    st.sidebar.title("Controls")

    if st.sidebar.button("Start"):
        build_index()
        st.sidebar.success("Done")

    index, paths = load_index()
    if index is None:
        st.info("Mount Folder !!!")
        return

    tab1, tab2 = st.tabs([" Text Query", " Image Query"])

    with tab1:
        st.subheader(" Search by Text")
        q = st.text_input("Type something (e.g., *red car*, *cat on sofa*):")
        k = st.slider("Results to show", 1, 30, 10, key="k_text")
        if st.button("Search", key="btn_text"):
            q_emb = embed_text(q).astype(np.float32)
            q_emb = q_emb / np.linalg.norm(q_emb)
            D, I = index.search(np.expand_dims(q_emb, 0), k)

            st.markdown("###  Results")
            cols = st.columns(5)
            for rank, idx in enumerate(I[0]):
                with cols[rank % 5]:
                    st.image(paths[idx], use_container_width=True)
                    st.caption(f"#{rank+1} • score={float(D[0][rank]):.3f}")

    with tab2:
        st.subheader(" Search by Image")
        up = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
        k2 = st.slider("Results to show", 1, 30, 10, key="k_img")
        if up and st.button("Search", key="btn_img"):
            img = Image.open(up).convert("RGB")
            st.image(img, caption="Query", width=250)
            q_emb = model.encode([img], convert_to_numpy=True)[0].astype(np.float32)
            q_emb = q_emb / np.linalg.norm(q_emb)
            D, I = index.search(np.expand_dims(q_emb, 0), k2)

            st.markdown("###  Results")
            cols = st.columns(5)
            for rank, idx in enumerate(I[0]):
                with cols[rank % 5]:
                    st.image(paths[idx], use_container_width=True)
                    st.caption(f"#{rank+1} • score={float(D[0][rank]):.3f}")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1ca97f, #0c3f79);
        min-height: 100vh;
        padding: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 8px rgba(229, 46, 113, 0.3);
        border: none;
        cursor: pointer;
        transition: background 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #e52e71, #ff8a00);
        box-shadow: 0 6px 12px rgba(229, 46, 113, 0.5);
    }
    .stButton > button:focus {
        outline: 2px solid #ff8a00;
        outline-offset: 2px;
    }
    .stSlider > div[data-baseweb="slider"] {
        padding: 0 12px;
    }
    .stSlider input[type="range"] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 6px;
        background: #ddd;
        outline: none;
        transition: background 0.3s ease;
        cursor: pointer;
    }
    .stSlider input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #e52e71;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(229, 46, 113, 0.6);
        transition: background 0.3s ease;
        margin-top: -8px; /* center thumb vertically */
    }
       .stSlider input[type="range"]:hover::-webkit-slider-thumb {
        background: #ff8a00;
        box-shadow: 0 4px 10px rgba(255, 138, 0, 0.7);
    }
    .stSlider input[type="range"]::-moz-range-thumb {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #e52e71;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(229, 46, 113, 0.6);
        transition: background 0.3s ease;
    }
    .stSlider input[type="range"]:hover::-moz-range-thumb {
        background: #ff8a00;
        box-shadow: 0 4px 10px rgba(255, 138, 0, 0.7);
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
