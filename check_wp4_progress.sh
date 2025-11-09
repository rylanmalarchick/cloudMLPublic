#!/bin/bash
echo "=================================="
echo "WP-4 Training Progress Check"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================="
echo

# Check if training is running
if ps aux | grep -v grep | grep wp4_cnn_model > /dev/null; then
    echo "✓ Training is RUNNING"
    echo
    
    # Show CPU usage
    ps aux | grep -v grep | grep wp4_cnn_model | awk '{print "  CPU: " $3 "%, Memory: " $4 "%"}'
    echo
    
    # Check for completed model files
    echo "Completed folds:"
    ls -1 sow_outputs/wp4_cnn/model_*.pth 2>/dev/null | wc -l | awk '{print "  " $1 " model files saved"}'
    echo
    
    # Check for reports
    echo "Completed modes:"
    for mode in image_only concat attention; do
        if [ -f "sow_outputs/wp4_cnn/WP4_Report_${mode}.json" ]; then
            echo "  ✓ ${mode}"
        else
            echo "  ⏳ ${mode} (in progress or pending)"
        fi
    done
else
    echo "✗ Training is NOT running"
    echo
    echo "Check results with:"
    echo "  python sow_outputs/wp4_final_summary.py"
fi

echo
echo "=================================="
