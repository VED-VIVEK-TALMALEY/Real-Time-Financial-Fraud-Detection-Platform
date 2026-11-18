from fastapi import FastAPI, HTTPException
from starlette.responses import Response
from .schemas import Transaction
from .scoring import score_payload

app = FastAPI(title="Fraud Scoring API")
MODEL_INFO = {"model": "fraud-xgb", "stage": "Production", "threshold": 0.5}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return MODEL_INFO


@app.post("/score")
def score(transaction: Transaction):
    try:
        payload = transaction.model_dump(by_alias=True)
        return score_payload(payload, threshold=float(MODEL_INFO["threshold"]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/reload")
def reload_model():
    MODEL_INFO["reloaded"] = True
    return {"status": "reloaded"}


@app.get("/metrics")
def metrics():
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response("# metrics unavailable\n", media_type="text/plain")
