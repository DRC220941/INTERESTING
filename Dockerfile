FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ajoutez cette ligne pour servir le frontend
RUN mkdir -p /app/static

CMD ["gunicorn", "-c", "gunicorn.conf.py", "-k", "uvicorn.workers.UvicornWorker", "src.main:app"]
