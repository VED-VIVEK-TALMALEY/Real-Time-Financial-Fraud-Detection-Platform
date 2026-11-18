# API

## GET /health

Returns service status.

## GET /model-info

Returns model name, registry stage, and score threshold.

## POST /score

Request:

```json
{"TransactionID": "123", "TransactionDT": 86400, "TransactionAmt": 42.5, "card1": "100"}
```

Response:

```json
{"transaction_id":"123","fraud_score":0.12,"xgb_score":0.10,"gnn_score":0.16,"decision":"review","latency_ms":1.2,"ring_id":null}
```
