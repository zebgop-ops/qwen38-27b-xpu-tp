#!/bin/bash
# Restarts vllm-qwen38 when it is "active" but not serving. Restart=on-failure never fires for a
# hang: we saw one startup where a GPU copy-engine fault ("Engine memory CAT error ... class=bcs")
# left a worker spinning in urEventWait forever. /health stays responsive during long prefills.
UNIT=${UNIT:-vllm-qwen38}
GRACE=${GRACE:-900}                  # seconds after (re)start before we judge (startup ~2.5 min)
URL=${URL:-http://localhost:8000/health}
systemctl is-active -q "$UNIT" || exit 0
since_us=$(systemctl show "$UNIT" -p ActiveEnterTimestampMonotonic --value)
up_s=$(cut -d. -f1 /proc/uptime)
age=$(( up_s - since_us / 1000000 ))
[ "$age" -lt "$GRACE" ] && exit 0
for _ in 1 2 3; do curl -sf -m 10 "$URL" >/dev/null && exit 0; sleep 20; done
logger -t vllm-watchdog "$UNIT active for ${age}s but $URL failed 3x; restarting"
systemctl restart "$UNIT"
