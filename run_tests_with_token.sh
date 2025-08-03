#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "🔑 Loading environment variables from .env file..."
    export $(cat .env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found"
    exit 1
fi

# Verify token is loaded
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ HUGGINGFACE_TOKEN not found in environment"
    exit 1
else
    echo "✅ HUGGINGFACE_TOKEN loaded: ${HUGGINGFACE_TOKEN:0:10}..."
fi

echo ""
echo "🧪 Running all tests with token..."
echo "=================================================="

# Run all candidates test
echo "📝 Running all candidates test..."
python test_all_candidates.py > test_all_candidates_results_with_token.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ All candidates test completed successfully"
else
    echo "❌ All candidates test failed"
fi

echo ""

# Run prompt improvements test
echo "📝 Running prompt improvements test..."
python test_prompt_improvements.py > test_prompt_improvements_results_with_token.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Prompt improvements test completed successfully"
else
    echo "❌ Prompt improvements test failed"
fi

echo ""

# Run temperature config test
echo "📝 Running temperature config test..."
python test_temperature_config.py > test_temperature_config_results_with_token.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Temperature config test completed successfully"
else
    echo "❌ Temperature config test failed"
fi

echo ""
echo "🏁 All tests completed!"
echo "📁 Results saved to:"
echo "   - test_all_candidates_results_with_token.txt"
echo "   - test_prompt_improvements_results_with_token.txt"
echo "   - test_temperature_config_results_with_token.txt" 