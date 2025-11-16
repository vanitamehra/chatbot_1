# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port (same as your FastAPI/Flask app)
EXPOSE 8000

# Command to run your app
# Replace main:app with your FastAPI app entry point
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
