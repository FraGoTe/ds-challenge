# syntax=docker/dockerfile:1.2
# Use the official Python slim image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose port 8080 (commonly used for web apps in containers)
EXPOSE 8080

# Command to run the FastAPI app with Gunicorn and Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "challenge.api:app"]