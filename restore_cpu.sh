#!/bin/bash
# Restore CPU to power-saving mode

echo "🔄 Restoring CPU to power-saving mode..."

# Check current state
echo "Current CPU governor:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

echo ""
echo "Restoring power-saving settings..."

# Set all cores to powersave governor (default)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "powersave" | sudo tee $cpu > /dev/null 2>/dev/null || true
done

# Restore minimum frequency to lowest setting
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    min_freq=$(cat ${cpu/min_freq/cpuinfo_min_freq} 2>/dev/null || echo "800000")
    echo $min_freq | sudo tee $cpu > /dev/null 2>/dev/null || true
done

# Re-enable turbo boost management
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>/dev/null || true

sleep 2

echo "✅ CPU restored to power-saving mode!"
echo ""
echo "New CPU governor:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

echo ""
echo "🌿 CPU is now optimized for battery life and thermal management."
echo "💡 Use ./boost_cpu.sh to re-enable performance mode when needed."
