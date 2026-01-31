import os
import numpy as np
import faiss
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from contextlib import asynccontextmanager

# ===================== CONFIG =====================
INDEX_FILE = "products.index"
FILENAMES_FILE = "filenames.npy"
BATCH_SIZE = 64

# ===================== GLOBAL STATE =====================
model = None
index = None
filenames = []

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, index, filenames

    print("Model yuklanmoqda...")
    model = SentenceTransformer('clip-ViT-B-32')
    model = model.to('cuda')
    print("Model tayyor! (GPU)")

    if os.path.exists(INDEX_FILE) and os.path.exists(FILENAMES_FILE):
        print("Index yuklanmoqda...")
        index = faiss.read_index(INDEX_FILE)
        filenames = list(np.load(FILENAMES_FILE))
        print(f"Yuklandi: {len(filenames)} ta rasm")

    yield

    print("Server to'xtamoqda...")


app = FastAPI(title="Asaxiy Image Search", lifespan=lifespan)

# ===================== SCHEMAS =====================
class InitRequest(BaseModel):
    folder: str

class AddRequest(BaseModel):
    path: str

class SearchRequest(BaseModel):
    path: str

class SearchResponse(BaseModel):
    result: str
    score: float

# ===================== FUNCTIONS =====================
def get_embedding(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return model.encode(img)


def get_embeddings_batch(image_paths: list):
    images = []
    valid_indices = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"Xato: {path} - {e}")
            continue

    if not images:
        return np.array([]), valid_indices

    embeddings = model.encode(images)
    return embeddings, valid_indices

# ===================== ENDPOINTS =====================
@app.post("/init")
def init_index(req: InitRequest):
    global index, filenames

    if not os.path.exists(req.folder):
        raise HTTPException(status_code=404, detail=f"Papka topilmadi: {req.folder}")

    all_files = os.listdir(req.folder)
    image_files = [f for f in all_files
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]

    if not image_files:
        raise HTTPException(status_code=400, detail="Papkada rasm topilmadi")

    print(f"Indexlash boshlanmoqda: {len(image_files)} ta rasm")

    all_vectors = []
    valid_filenames = []

    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Indexing"):
        batch_files = image_files[i:i+BATCH_SIZE]
        batch_paths = [os.path.join(req.folder, f) for f in batch_files]

        vectors, valid_indices = get_embeddings_batch(batch_paths)

        if len(vectors) > 0:
            all_vectors.append(vectors)
            for idx in valid_indices:
                valid_filenames.append(batch_files[idx])

    if not all_vectors:
        raise HTTPException(status_code=400, detail="Hech qaysi rasm o'qilmadi")

    vectors_np = np.vstack(all_vectors).astype('float32')
    faiss.normalize_L2(vectors_np)

    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)

    filenames = [os.path.join(req.folder, f) for f in valid_filenames]

    faiss.write_index(index, INDEX_FILE)
    np.save(FILENAMES_FILE, np.array(filenames))

    return {
        "status": "ok",
        "indexed": len(filenames)
    }


@app.post("/add")
def add_image(req: AddRequest):
    global index, filenames

    if index is None:
        if os.path.exists(INDEX_FILE) and os.path.exists(FILENAMES_FILE):
            index = faiss.read_index(INDEX_FILE)
            filenames = list(np.load(FILENAMES_FILE))
        else:
            raise HTTPException(status_code=400, detail="Index mavjud emas. Avval /init qiling")

    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"Rasm topilmadi: {req.path}")

    vec = get_embedding(req.path)
    vec = np.array(vec).reshape(1, -1).astype('float32')
    faiss.normalize_L2(vec)

    index.add(vec)
    filenames.append(req.path)

    faiss.write_index(index, INDEX_FILE)
    np.save(FILENAMES_FILE, np.array(filenames))

    return {
        "status": "ok",
        "total": len(filenames)
    }


@app.post("/search", response_model=SearchResponse)
def search_image(req: SearchRequest):
    global index, filenames

    if index is None:
        if os.path.exists(INDEX_FILE) and os.path.exists(FILENAMES_FILE):
            index = faiss.read_index(INDEX_FILE)
            filenames = list(np.load(FILENAMES_FILE))
        else:
            raise HTTPException(status_code=400, detail="Index mavjud emas. Avval /init qiling")

    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"Rasm topilmadi: {req.path}")

    vec = get_embedding(req.path)
    vec = np.array(vec).reshape(1, -1).astype('float32')
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, 1)

    best_idx = indices[0][0]
    best_score = float(scores[0][0])

    return SearchResponse(
        result=filenames[best_idx],
        score=best_score
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed": len(filenames) if filenames else 0,
        "gpu": "cuda" if model else "not loaded"
    }


@app.get("/")
def root():
    return {"message": "Asaxiy Image Search API", "docs": "/docs"}
