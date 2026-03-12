FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (tools + scripts; model copied separately or mounted)
COPY tools/ tools/
COPY scripts/ scripts/

# Copy model(s); ensure models/rent_price_model.joblib exists before building, or mount at runtime
ENV MODEL_PATH=models/rent_price_model.joblib
COPY models/ models/

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "scripts.serve:app"]
