# Asaxiy Image Search API

CLIP + FAISS asosida rasm bo'yicha qidirish tizimi.

## Texnologiyalar

| Texnologiya | Vazifasi |
|-------------|----------|
| CLIP | Rasmlardan embedding olish |
| FAISS | Vektor qidirish |
| FastAPI | REST API |
| RTX 4070 Super | GPU tezlashtirish |

## O'rnatish

```bash
pip install -r requirements.txt
```

## Ishga tushirish

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /init
Papkadagi barcha rasmlarni indexlash.

```bash
curl -X POST http://localhost:8000/init \
  -H "Content-Type: application/json" \
  -d '{"folder": "/data/products"}'
```

**Response:**
```json
{"status": "ok", "indexed": 300000}
```

### POST /add
Yangi rasmni indexga qo'shish.

```bash
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/products/new_item.jpg"}'
```

**Response:**
```json
{"status": "ok", "total": 300001}
```

### POST /search
Rasmga o'xshashini topish.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/query/image.jpg"}'
```

**Response:**
```json
{"result": "/data/products/similar.jpg", "score": 0.89}
```

### GET /health
Server holatini tekshirish.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok", "indexed": 300000, "gpu": "cuda"}
```

## Tezlik

| Operatsiya | Vaqt |
|------------|------|
| /init (300k rasm) | ~30-60 daqiqa |
| /add (1 rasm) | ~50 ms |
| /search | ~10 ms |

## Fayllar

```
├── main.py           # API server
├── requirements.txt  # Kutubxonalar
├── products.index    # FAISS index (auto-generated)
└── filenames.npy     # Rasm nomlari (auto-generated)
```
