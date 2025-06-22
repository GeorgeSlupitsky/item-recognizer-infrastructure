#!/bin/bash
set -e

PROJECT_ID=${1:-2}  # Default project ID is 2
OUTPUT_FILE="label_studio_export.json"  # Default output filename

if [ -z "$REFRESH_TOKEN" ]; then
  echo "Error: REFRESH_TOKEN environment variable is not set."
  echo "Usage: export REFRESH_TOKEN=your_token"
  echo "Then run: ./export_json.sh [project_id]"
  exit 1
fi

echo "Getting access token..."
ACCESS_TOKEN=$(curl -s -X POST http://localhost:8080/api/token/refresh \
  -H "Content-Type: application/json" \
  -d "{\"refresh\": \"$REFRESH_TOKEN\"}" | grep -o '"access":"[^"]*"' | cut -d'"' -f4)

if [ -z "$ACCESS_TOKEN" ]; then
  echo "Error: Failed to get access token. Check your refresh token and Label Studio availability."
  exit 1
fi

echo "Exporting data from project #${PROJECT_ID} in JSON format..."
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "http://localhost:8080/api/projects/$PROJECT_ID/export?exportType=JSON" \
  -o "$OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
  echo "Export successful! File saved as $OUTPUT_FILE"
else
  echo "Error: Export failed or no data was exported."
  exit 1
fi

echo "Done. You can now use the exported data for training, analysis, or conversion."
