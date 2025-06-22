#!/bin/bash
# Smart server startup with CPU governor management

cd "$(dirname "$0")"

# Save original CPU governor state
echo "💾 Saving original CPU governor state..."
ORIGINAL_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
echo "Original governor: $ORIGINAL_GOVERNOR"
echo "$ORIGINAL_GOVERNOR" > .original_governor

# Function to restore CPU governor on exit
restore_cpu() {
    if [ -f ".original_governor" ]; then
        RESTORE_TO=$(cat .original_governor)
        echo ""
        echo "🔄 Restoring CPU governor to: $RESTORE_TO"
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "$RESTORE_TO" | sudo tee $cpu > /dev/null 2>/dev/null || true
        done
        rm -f .original_governor
        echo "✅ CPU governor restored"
    fi
}

# Set trap to restore CPU on script exit (Ctrl+C, normal exit, etc.)
trap restore_cpu EXIT INT TERM

# Boost CPU performance
echo "⚡ Boosting CPU performance..."
sudo ./boost_cpu.sh

# Start the server
echo ""
echo "🚀 Starting NPGlue server..."
echo "Press Ctrl+C to stop server and restore CPU governor"
echo ""

source npglue-env/bin/activate
python server_production.py

# restore_cpu will be called automatically by the trap
