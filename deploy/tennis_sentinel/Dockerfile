FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY sports_edge /app/sports_edge
COPY scripts /app/scripts
RUN useradd -m --uid 10001 observer
# Cloud volumes mount after image build, often owned by root; create and
# assign only the dedicated worker ledger directory on startup.
COPY deploy/tennis_sentinel/start.sh /app/start.sh
RUN chmod 755 /app/start.sh
ENTRYPOINT ["/app/start.sh"]
