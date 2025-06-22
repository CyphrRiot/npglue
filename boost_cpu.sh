#!/bin/bash
# Maximize CPU performance for LLM inference

echo "🚀 Optimizing CPU for maximum LLM performance..."

# Check current state
echo "Current CPU governor:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

echo "Current CPU frequencies:"
cat /proc/cpuinfo | grep "cpu MHz" | head -4

echo ""
echo "Setting CPU to performance mode..."

# Set all cores to performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee $cpu > /dev/null
done

# Disable CPU idle states for maximum performance  
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>/dev/null || true
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>/dev/null || true

# Set maximum CPU frequency
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    max_freq=$(cat ${cpu/min/max})
    echo $max_freq | sudo tee $cpu > /dev/null 2>/dev/null || true
done

sleep 2

echo "✅ CPU optimization complete!"
echo ""
echo "New CPU governor:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

echo "New CPU frequencies:"
cat /proc/cpuinfo | grep "cpu MHz" | head -4

echo ""
echo "🔥 CPU should now be running at maximum performance!"
echo "Run your LLM tests now for significantly better speed."
