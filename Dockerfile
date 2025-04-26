# Pull base image
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    wget \
    ca-certificates \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6 \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip show marshmallow
RUN pip install --no-cache-dir --force-reinstall pillow

# Download NLTK data (only once Pillow and NLTK are installed)
RUN python -m nltk.downloader \
        punkt \
        punkt_tab \
        averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng \
        stopwords \
    && mkdir -p /usr/share/nltk_data \
    && mv /root/nltk_data/* /usr/share/nltk_data

# Copy project
COPY . .
