FROM python:3.10-slim

# HF Spaces requires non-root user with uid 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Healthcheck so HF Space knows when the server is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["python", "server/app.py"]
