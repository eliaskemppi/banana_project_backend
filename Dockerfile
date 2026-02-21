# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (optimizes build speed)
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and saved_models into the container
COPY . .

# 6. Tell Docker which port the app runs on
EXPOSE 8000

# 7. The command to start your FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]