# Image/Text Search (CLIP + FAISS + Streamlit)

Upload a folder of images, build a FAISS index with CLIP embeddings, and search by image or text.

## Run with Docker
```bash
docker build -t image_search .
docker run --rm -p 8501:8501 -v $(pwd)/gallery:/app/gallery image_search
```
Then open http://localhost:8501

## Local (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```
