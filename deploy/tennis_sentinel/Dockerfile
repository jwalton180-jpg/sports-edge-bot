FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY sports_edge /app/sports_edge
COPY scripts /app/scripts
RUN useradd -m --uid 10001 observer
USER observer
CMD ["python", "-m", "scripts.tennis_realtime_sentinel"]
