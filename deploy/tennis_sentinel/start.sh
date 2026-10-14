#!/bin/sh
set -eu

: "${SPORTSEDGE_SENTINEL_DATA_DIR:=/data/tennis}"
case "${SPORTSEDGE_SENTINEL_DATA_DIR}" in
  /data/*) ;;
  *) echo "Sentinel must use the isolated /data Railway volume" >&2; exit 2 ;;
esac

# Only the short initialization step runs as root. All network polling,
# Kalshi read-only streaming and ntfy publishing run as UID 10001.
install -d -m 0700 -o observer -g observer "${SPORTSEDGE_SENTINEL_DATA_DIR}"
if [ "$(id -u)" -ne 0 ]; then
  echo "Sentinel volume initializer requires container root" >&2
  exit 2
fi
exec su -s /bin/sh observer -c 'exec python -m scripts.tennis_realtime_sentinel'
