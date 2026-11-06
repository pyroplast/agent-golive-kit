FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deploykit ./deploykit
COPY kb ./kb
COPY suite ./suite
COPY config ./config
CMD ["python", "-m", "deploykit.job"]
