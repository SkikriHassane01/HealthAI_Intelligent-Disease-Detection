FROM python:3.10-slim

EXPOSE 8000

WORKDIR /app

COPY requirements.txt .

# Install build tools, gcc, and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Define environment variable
ENV NAME healthai

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]