#!/bin/bash
# quick_debug.sh - One command to run tests and extract errors

echo "🔬 Running tests and extracting error information..."
echo ""

python run_tests.py 2>&1 | grep -A3 -B1 -E "(^FAILED |^E   |^>.*assert|\.py:[0-9]+: (RuntimeError|AssertionError|ValueError|TypeError))" | \
sed 's/^FAILED /\n❌ FAILED TEST: /' | \
sed 's/^E   /   🚨 ERROR: /' | \
sed 's/^>/   📍 CODE: /' | \
sed '/^--$/d' | \
head -50

echo ""
echo "🔍 Copy the above condensed error info for debugging"
