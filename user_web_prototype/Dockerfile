FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py backend.py ./
COPY static ./static

ENV USER_WEB_DATA=/data
EXPOSE 8503
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8503"]
