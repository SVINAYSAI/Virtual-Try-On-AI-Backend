# ---- 1. Use an official lightweight Python image ----
FROM python:3.11-slim

# ---- 2. Set working directory inside the container ----
WORKDIR /app

# ---- 3. Install system dependencies (opencv and rembg dependencies) ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- 4. Copy requirements first for caching ----
COPY requirements.txt .

# ---- 5. Install Python dependencies ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- 6. Copy the FastAPI app ----
COPY main.py .
COPY config.py .
COPY __init__.py .
COPY gemini_API/ ./gemini_API/
COPY image_apis/ ./image_apis/
COPY filter_API/ ./filter_API/

# ---- 6.1. Copy Google Cloud credentials file ----
COPY marine-set-447307-k6-d2077b260b04.json .

# ---- 6.2. Create outputs directory ----
RUN mkdir -p outputs

# ---- 7. Expose FastAPI port ----
EXPOSE 8000

# ---- 8. Start FastAPI using uvicorn ----
# Note: Removed --reload for production; use it only in development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]