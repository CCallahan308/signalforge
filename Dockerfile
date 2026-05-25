# SignalForge Dockerfile
# Pinned to 3.11 to match .python-version, pyproject (requires-python >=3.11)
# and CI. The numpy/scipy/scikit-learn pins do not have wheels for 3.14.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (dashboard + pipeline only; the full stack in
# requirements-full.txt is aspirational and mostly unused at runtime)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
