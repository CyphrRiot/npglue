#!/bin/bash
cd "$(dirname "$0")"

# Save original CPU governor state
echo "📊 Saving original CPU performance settings..."
ORIGINAL_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "powersave")
echo "   Original governor: $ORIGINAL_GOVERNOR"

# Function to restore CPU settings on exit
cleanup() {
    echo ""
    echo "🔄 Restoring CPU settings..."
    
    # Restore original governor
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "$ORIGINAL_GOVERNOR" | sudo tee $cpu > /dev/null 2>/dev/null || true
    done
    
    # Reset CPU frequency scaling
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        min_freq=$(cat ${cpu/min_freq/cpuinfo_min_freq} 2>/dev/null || echo "800000")
        echo $min_freq | sudo tee $cpu > /dev/null 2>/dev/null || true
    done
    
    echo "✅ CPU settings restored to: $ORIGINAL_GOVERNOR"
    echo "👋 NPGlue server stopped cleanly"
}

# Set up signal handlers
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM

source npglue-env/bin/activate

echo "🚀 Starting NPGlue server..."
echo "💡 Press Ctrl+C to stop server and restore CPU settings"
echo ""

python server_production.py
