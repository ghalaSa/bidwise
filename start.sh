#!/bin/bash
# Startup script for BidWise AI Proposal Reviewer Backend & Frontend

PORT=8080
echo "============================================="
echo "  BidWise AI SaaS Web App - Backend Server   "
echo "============================================="
echo "Starting Flask Server on port $PORT..."
echo "Open your browser and navigate to:"
echo "👉 http://localhost:$PORT"
echo "============================================="
echo "Press Ctrl+C to stop the server."

python3 server.py
