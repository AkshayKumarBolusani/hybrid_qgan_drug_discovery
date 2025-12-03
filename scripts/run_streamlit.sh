#!/bin/bash
# Launch Streamlit UI

echo "======================================"
echo "Launching HQ-GAN Drug Discovery UI"
echo "======================================"
echo ""
echo "🧬 Starting Streamlit application..."
echo ""
echo "Access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")/.."

streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address localhost
