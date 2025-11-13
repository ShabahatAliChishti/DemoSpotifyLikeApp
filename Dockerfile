FROM python:3.9-slim

# Install dependencies

RUN pip install --no-cache-dir \
    pandas==2.1.0 \
    numpy==1.26.0 \
    xgboost==1.7.6 \
    scikit-learn==1.3.1 \
    scipy==1.13.1

# Set working dir
WORKDIR /app
