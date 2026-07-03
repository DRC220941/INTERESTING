# Utilise une image Python avec Node.js pour construire le frontend (si besoin plus tard)
FROM python:3.11-slim

WORKDIR /app

# Installe les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le projet (y compris le dossier static)
COPY . .

# Active CORS et monte le dossier static
CMD ["gunicorn", "-c", "gunicorn.conf.py", "-k", "uvicorn.workers.UvicornWorker", "src.main:app"]
