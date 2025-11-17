# 1. Base image
FROM python:3.10-slim

# 2. Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working directory
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    pkg-config \
    libcairo2-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements
COPY requirements.txt .

# 6. Upgrade pip and install Torch CPU first
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1

# 7. Install remaining dependencies (all pinned versions)
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy app code
COPY . .

# 9. Expose port
EXPOSE 8000

# 10. Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
