# 1. Base image with Python
FROM python:3.9-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy requirements and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app code into the image
COPY dashboard/ ./dashboard
COPY scripts/  ./scripts

# 4b. Copy processed data
COPY data/processed ./data/processed

# 5. Expose the port Streamlit uses
EXPOSE 8501

# 6. Launch Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["dashboard/04_dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.enableCORS=false"]

