FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the current project structure explicitly.
COPY src/ /app/src/
COPY data/ /app/data/
COPY main.py /app/main.py
COPY generate_data.py /app/generate_data.py
COPY README.md /app/README.md
COPY RAPPORT_TECHNIQUE.md /app/RAPPORT_TECHNIQUE.md

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
