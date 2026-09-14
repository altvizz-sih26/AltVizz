FROM python:3.11-slim

# System libraries needed by rasterio (GDAL) and opencv
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT automatically; default to 8000 for local testing
ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT