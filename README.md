# HackRx 6.0 Retrieval System

## Local Development

1. **Clone the repository**
```bash
git clone <your-repo>
cd hackrx-retrieval-system
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run locally**
```bash
uvicorn main:app --reload --port 8000
```

5. **Test the API**
```bash
python test_api.py
```

## Docker Deployment

1. **Build Docker image**
```bash
docker build -t hackrx-retrieval .
```

2. **Run Docker locally**
```bash
docker run -p 8000:8000 --env-file .env hackrx-retrieval
```

## Railway Deployment

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login to Railway**
```bash
railway login
```

3. **Create new project**
```bash
railway init
```

4. **Set environment variables**
```bash
railway variables set PINECONE_API_KEY=your_key
railway variables set PINECONE_HOST=your_env
railway variables set PINECONE_INDEX_NAME=hackrx-index
railway variables set GEMINI_API_KEY=your_key
railway variables set BEARER_TOKEN=ea7e09c77954bb70be43b33d0410aee6cdb82253847f23f963a30e2f0145f771
```

5. **Deploy**
```bash
railway up
```

6. **Get deployment URL**
```bash
railway open
```

## API Usage

```bash
curl -X POST https://your-app.railway.app/api/v1/hackrx/run \
  -H "Authorization: Bearer ea7e09c77954bb70be43b33d0410aee6cdb82253847f23f963a30e2f0145f771" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["Question 1?", "Question 2?"]
  }'
```

## Monitoring

- Health check: GET /health
- API docs: GET /docs (FastAPI automatic documentation)