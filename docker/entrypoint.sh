

set -e

echo "[entrypoint] MODE=${MODE}"

if [ "$MODE" = "api" ]; then
  echo "[entrypoint] starting API server..."
  python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

elif [ "$MODE" = "batch" ]; then
  echo "[entrypoint] running batch forecast..."
  python -m src.pipelines.batch_forecast --horizons 24 --out_dir artifacts/forecasts

else
  echo "Unknown MODE: ${MODE} (use 'api' or 'batch')"
  exit 1
fi