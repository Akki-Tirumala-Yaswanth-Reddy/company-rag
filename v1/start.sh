#!/bin/sh
set -e
# AWS sets AWS_LAMBDA_FUNCTION_NAME inside Lambda. On Lambda only /tmp is writable,
# so we work on a copy of the data there. Locally / on Cloud Run this block is skipped.
if [ -n "$AWS_LAMBDA_FUNCTION_NAME" ]; then
  cp -r /app/chroma_db /tmp/chroma_db
  cp -r /app/data /tmp/data
  export CHROMA_DIR=/tmp/chroma_db DB_PATH=/tmp/data/acme.db
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"