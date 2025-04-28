# Use an official lightweight Python image.
FROM python:3.12-slim

# Ensure output is not buffered and set the working directory.
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install system build dependencies (if needed for some packages)
# Uncomment the following if you run into build issues:
# RUN apt-get update && apt-get install -y gcc libpq-dev

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port your application will run on. (Change if your app uses a different port.)
EXPOSE 8000

# Set environment variables (if needed, such as API keys)
# Use Docker secrets or ENV commands to securely inject these if required.
# ENV PINECONE_API=pcsk_4bLR9o_3crxHE9zjHW76VdRnBPi2Xo794pQnKSifnRfQ9iQc6U3iqeqeyVEZ3RjBPYtoD4

# Define the command to run your application.
# Here we assume a FastAPI app inside deployment_code/app.py. Adjust the command as needed.
CMD ["python", "-m", "uvicorn", "deployment_code.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]