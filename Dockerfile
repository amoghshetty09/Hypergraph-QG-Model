# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy your code to container
COPY . /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Set environment variable for offline use or Transformers cache
ENV TRANSFORMERS_CACHE=/app/cache

# Run evaluation script on container start
CMD ["python", "evaluate.py"]
