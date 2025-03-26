# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY backend/requirements.txt .


# Copy .env file
COPY backend/.env .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY backend/app/ ./app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]