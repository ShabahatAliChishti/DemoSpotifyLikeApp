FROM python:3.9-slim

# Install basic system tools for FAISS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (versions compatible with Python 3.9)
RUN pip install --no-cache-dir \
    pandas==2.1.0 \
    numpy==1.26.0 \
    xgboost==1.7.6 \
    scikit-learn==1.3.1 \
    scipy==1.13.1 \
    faiss-cpu==1.7.4 \
    kfp==2.14.6

# Set working directory
WORKDIR /app

# Default command (optional)
CMD ["python3"]
