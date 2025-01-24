# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the Flask app port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "main.py"]
