Pulse NLP Microservices
Overview
A powerful financial NLP API leveraging AI-driven sentiment and market impact analysis, optimized by vector store caching with Chromadb and the Gemini AI platform. Uses local SentenceTransformer embeddings (all-MiniLM-v2) for cost-effective, real-time processing.

Features
Advanced financial sentiment and impact analysis

Real-time and batch (bulk) processing

Intelligent caching of analysis results in Chromadb vector DB

OpenAPI compliant endpoints

Linear flattened JSON output

API Endpoints
POST /api/v1/analyze
Analyze a single financial statement

Input: statement (text), author info, platform, engagement metrics, optional market & user profile

Output: detailed flattened analysis with sentiment, market impact, assets, reasoning, risks

POST /api/v1/analyze/bulk
Analyze multiple statements in batch

Input: list of statement objects (same shape as single analyze)

Output: list of flattened analysis, same fields as single analyze

GET /health
Health & dependency status (flattened JSON)

GET /stats
Service usage & performance metrics

Running

# 1. Clone and setup
git clone <your-repo>
cd Pulse-nlp-microservices
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Configure environment
# Edit .env file with your GEMINI_API_KEY

# 4. Run service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


Or via Docker:

# Configure .env first, then:
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop service
docker-compose down
