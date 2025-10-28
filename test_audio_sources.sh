#!/bin/bash

# 🔊 Advanced EchoPortal Audio Diagnostic Tool
# Actually performs manual parec testing to find Omarchy's audio routing

echo "🔍 Advanced EchoPortal Audio Diagnostic Tool"
echo "============================================"
echo ""

echo "🎵 Checking PulseAudio status..."
if ! pgrep -x "pulseaudio" > /dev/null && ! pgrep -x "pipewire-pulse" > /dev/null; then
    echo "❌ ERROR: PulseAudio/PipeWire not running!"
    exit 1
fi
echo "✅ PulseAudio/PipeWire is running"
echo ""

echo "🎭 Current audio setup:"
echo "   Default Sink: $(pactl get-default-sink)"
echo "   Default Source: $(pactl get-default-source)"
echo ""

echo "🎸 IMPORTANT: Make sure Omarchy is PLAYING MUSIC right now!"
echo "            This tests actual audio capture capability."
echo ""

# Function to test an audio device with parec
test_audio_device() {
    local device="$1"
    local name="$2"

    echo "🎧 Testing: $name"
    echo "   Device: $device"

    # Run parec for 3 seconds, capture and analyze output
    echo "   🔊 Listening for 3 seconds..."
    echo "   (Press Ctrl+C if audio bars don't appear)"

    # Try to capture audio and look for non-zero activity
    # parec sends output to stderr, so we capture that to get audio visualization
    timeout 3 parec --device="$device" --format=s16le --rate=44100 /dev/null 2>&1 || true

    echo "   📊 If you saw moving audio bars above, AUDIO IS DETECTED!"
    echo "   🎯 If you saw only empty bars, no audio on this device."
    echo ""
}

echo "🎚️  Testing HDMI monitor (most likely for Omarchy):"
test_audio_device "alsa_output.pci-0000_00_1f.3.hdmi-stereo.monitor" "HDMI Output Monitor"

echo "🎚️  Testing system default sink monitor:"
default_sink=$(pactl get-default-sink)
test_audio_device "$default_sink.monitor" "Default Sink Monitor"

echo "🎯 Testing all available sources:"
echo "---------------------------------"
pactl list sources short | while read -r line; do
    if [[ -n "$line" ]]; then
        device_name=$(echo "$line" | awk '{print $1}')
        device_desc=$(echo "$line" | cut -d' ' -f2-)

        echo "🎚️  Testing: $device_desc"
        test_audio_device "$device_name" "$device_desc"
    fi
done

echo "🎉 Diagnostic Complete!"
echo ""
echo "🎯 IF YOU SAW MOVING AUDIO BARS on any device:"
echo "   That device captures Omarchy's audio - note the device name!"
echo ""
echo "🎯 IF NO DEVICES SHOWED AUDIO BARS:"
echo "   Make sure Omarchy is actually playing music, then run again."
echo ""
echo "💡 Next: Modify EchoPortal to prioritize the working device!"
echo "   Copy the device name that showed audio bars."
