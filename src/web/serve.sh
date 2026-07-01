#!/bin/bash
# Start the HyQMOM viewer server. Usage: ./serve.sh [port]
PORT=${1:-8000}
cd "$(dirname "$0")"
echo ""
echo "  HyQMOM viewer serving on port $PORT"
echo "  ---------------------------------------------"
echo "  On your laptop:   ssh -L $PORT:localhost:$PORT phoenix"
echo "  Then open:        http://localhost:$PORT/viewer.html"
echo "  (Ctrl-C to stop)"
echo ""
python3 -m http.server $PORT --bind 127.0.0.1
