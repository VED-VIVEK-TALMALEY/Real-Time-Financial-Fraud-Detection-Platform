#!/usr/bin/env bash
set -euo pipefail
curl --fail --max-time 10 http://localhost:8000/health
curl --fail --max-time 10 -H 'Content-Type: application/json' \
  -d '{"TransactionID":"smoke","TransactionAmt":10,"card1":"1"}' \
  http://localhost:8000/score
