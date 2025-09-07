# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
