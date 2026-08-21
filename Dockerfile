# Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first (optimizes build speed)
COPY requirements.txt .

# Install dependencies

# First, install the light-weight CPU versions of torch
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and saved_models into the container
COPY . .

# Define the port the app will run on
EXPOSE 8000

# The command to start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]