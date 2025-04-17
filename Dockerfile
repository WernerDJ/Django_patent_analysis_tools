# Pull base image
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

RUN apt-get update \
    && apt-get install -y gcc libpq-dev build-essential wget ca-certificates \
    && apt-get clean

# Install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (including all required English language-specific models)
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
