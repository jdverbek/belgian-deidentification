#!/bin/bash

# Render build script for Belgian Document Deidentification System

set -e

echo "🏗️  Starting Render build process..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    python3-dev

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install the application in development mode
echo "🔧 Installing application..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{input,output,models,logs}
mkdir -p /tmp/belgian_deidentification

# Download and cache models (if needed)
echo "🤖 Preparing NLP models..."
python -c "
import os
import logging
logging.basicConfig(level=logging.INFO)

try:
    # Pre-download models to avoid cold start delays
    from transformers import AutoTokenizer, AutoModel
    
    model_name = 'DTAI-KULeuven/robbert-2023-dutch-large'
    print(f'Downloading {model_name}...')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print('✅ Models downloaded successfully')
    
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
    print('Models will be downloaded on first use')
"

# Set permissions
echo "🔐 Setting permissions..."
chmod -R 755 data/
chmod +x src/belgian_deidentification/api/main.py

echo "✅ Build completed successfully!"

