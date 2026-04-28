# Palestine SME Social Media Intelligence Platform

Full-stack analytics platform for Palestinian SMEs' organic social media posts. Upload a CSV dataset, validate/clean it, engineer KPIs, run data mining models (clustering, PCA, association rules, time-series trends, anomalies, network analysis), and explore results in a React dashboard.

## Stack
- Backend: Python 3.11+, FastAPI, Pandas, NumPy, scikit-learn, mlxtend, NetworkX, pydantic
- Frontend: React + TypeScript + Vite, Tailwind CSS, Recharts, Axios, React Router
- Storage: local filesystem under `backend/storage/`

## Quickstart (local dev)

### Backend
```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:
- http://localhost:8000
- http://localhost:8000/docs

### Frontend
```powershell
cd .\frontend
npm install
npm run dev
```

Frontend URL:
- http://localhost:5173

## Sample data
Use `data/sample_synthetic_posts.csv` to test uploads and the pipeline.

## Docs
See `instructions.md` for full details (pipeline steps, output files, API list, dashboard pages, team split, timeline, troubleshooting, and deliverables checklist).

