#!/bin/bash
# Smoke test script for MarsHab
# Runs a small end-to-end scenario to verify system functionality

echo "========================================"
echo "  MarsHab Smoke Test"
echo "========================================"
echo ""

set -e

# Test 1: Health check
echo "Test 1: Health Check"
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/v1/health/live)
if [ "$HEALTH_RESPONSE" -eq 200 ]; then
    echo "  ✓ Health check passed"
else
    echo "  ✗ Health check failed with status $HEALTH_RESPONSE"
    exit 1
fi

# Test 2: List presets
echo "Test 2: List Presets"
PRESETS_RESPONSE=$(curl -s http://localhost:5000/api/v1/analysis/presets)
SITE_PRESETS_COUNT=$(echo "$PRESETS_RESPONSE" | jq '.site_presets | length')
if [ "$SITE_PRESETS_COUNT" -gt 0 ]; then
    echo "  ✓ Found $SITE_PRESETS_COUNT site presets"
else
    echo "  ✗ No presets found"
    exit 1
fi

# Test 3: Example ROIs
echo "Test 3: Example ROIs"
EXAMPLES_RESPONSE=$(curl -s http://localhost:5000/api/v1/examples/rois)
EXAMPLES_COUNT=$(echo "$EXAMPLES_RESPONSE" | jq 'length')
if [ "$EXAMPLES_COUNT" -gt 0 ]; then
    echo "  ✓ Found $EXAMPLES_COUNT example ROIs"
else
    echo "  ✗ No examples found"
    exit 1
fi

echo ""
echo "========================================"
echo "  All smoke tests passed!"
echo "========================================"

