#!/bin/bash

# ════════════════════════════════════════════════════════════════
# CSV Upload Performance Test Script
# Tests the complete flow: CSV → Server → Processing → Response
# ════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          📊 CSV UPLOAD PERFORMANCE TEST                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if server is running
echo "🔍 Checking server status..."
if ps aux | grep "api_server" | grep -v grep > /dev/null; then
    echo "✅ Server is running"
    SERVER_PID=$(ps aux | grep "api_server" | grep -v grep | awk '{print $2}')
    echo "   Process ID: $SERVER_PID"
else
    echo "❌ Server is not running!"
    echo "   Start it with: nohup python -m server.api_server --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "TEST 1: Upload Small CSV (400 clients)"
echo "═══════════════════════════════════════════════════════════════════════"

CSV_FILE="data/LVMH_Sales_Database.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ CSV file not found: $CSV_FILE"
    exit 1
fi

echo "📁 File: $CSV_FILE"
echo "📏 Size: $(du -h "$CSV_FILE" | cut -f1)"
echo "📝 Lines: $(wc -l < "$CSV_FILE") records"
echo ""
echo "⏱️  Starting upload test..."
echo ""

# Upload CSV and measure time
START=$(date +%s.%N)

RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" \
  -X POST \
  -F "file=@${CSV_FILE}" \
  http://localhost:8000/api/upload-csv)

END=$(date +%s.%N)

HTTP_CODE=$(echo "$RESPONSE" | tail -n 2 | head -n 1)
CURL_TIME=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | head -n -2)

ELAPSED=$(echo "$END - $START" | bc)

echo "📤 Upload Response:"
echo "   HTTP Status: $HTTP_CODE"
echo "   Upload Time: ${CURL_TIME}s"
echo "   Total Time: ${ELAPSED}s"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Upload successful!"
    echo ""
    echo "📊 Response Data:"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
else
    echo "❌ Upload failed!"
    echo "Response: $BODY"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "TEST 2: Get ML Predictions"
echo "═══════════════════════════════════════════════════════════════════════"

START=$(date +%s.%N)
PREDICTIONS=$(curl -s http://localhost:8000/api/predictions)
END=$(date +%s.%N)
PRED_TIME=$(echo "$END - $START" | bc)

echo "⏱️  Fetch Time: ${PRED_TIME}s"
echo ""

PRED_COUNT=$(echo "$PREDICTIONS" | jq '. | length' 2>/dev/null || echo "0")
echo "🤖 ML Predictions Available: $PRED_COUNT clients"

if [ "$PRED_COUNT" -gt 0 ]; then
    echo ""
    echo "📊 Sample Prediction:"
    echo "$PREDICTIONS" | jq '.[0]' 2>/dev/null | head -20
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "TEST 3: Get Dashboard Data"
echo "═══════════════════════════════════════════════════════════════════════"

START=$(date +%s.%N)
DASHBOARD=$(curl -s http://localhost:8000/api/dashboard-data)
END=$(date +%s.%N)
DASH_TIME=$(echo "$END - $START" | bc)

echo "⏱️  Fetch Time: ${DASH_TIME}s"
echo ""

CLIENT_COUNT=$(echo "$DASHBOARD" | jq '.clients | length' 2>/dev/null || echo "0")
SEGMENT_COUNT=$(echo "$DASHBOARD" | jq '.segments | length' 2>/dev/null || echo "0")
CONCEPT_COUNT=$(echo "$DASHBOARD" | jq '.concepts | length' 2>/dev/null || echo "0")

echo "📊 Dashboard Data:"
echo "   • Clients: $CLIENT_COUNT"
echo "   • Segments: $SEGMENT_COUNT"
echo "   • Concepts: $CONCEPT_COUNT"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📈 PERFORMANCE SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "⏱️  API Response Times:"
echo "   • CSV Upload:         ${ELAPSED}s"
echo "   • Get Predictions:    ${PRED_TIME}s"
echo "   • Get Dashboard:      ${DASH_TIME}s"
TOTAL_TIME=$(echo "$ELAPSED + $PRED_TIME + $DASH_TIME" | bc)
echo "   • Total:              ${TOTAL_TIME}s"

echo ""
echo "🚀 Throughput:"
if [ "$CLIENT_COUNT" -gt 0 ] && [ $(echo "$ELAPSED > 0" | bc) -eq 1 ]; then
    THROUGHPUT=$(echo "scale=2; $CLIENT_COUNT / $ELAPSED" | bc)
    echo "   • Clients/second: $THROUGHPUT"
fi

echo ""
echo "💾 Data Generated:"
echo "   • Client Profiles: $CLIENT_COUNT"
echo "   • Segments: $SEGMENT_COUNT"
echo "   • Concepts Extracted: $CONCEPT_COUNT"
echo "   • ML Predictions: $PRED_COUNT"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ TEST COMPLETE!                               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Server URLs:"
echo "   • API: http://localhost:8000"
echo "   • Swagger: http://localhost:8000/docs"
echo "   • Dashboard: http://localhost:5173 (if running)"
echo ""
