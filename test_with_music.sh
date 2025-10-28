#!/bin/bash

# 🔊 EchoPortal Music Testing Script
# Tests if EchoPortal can capture audio when music is actually playing

echo "🎵 Testing EchoPortal Audio Capture WITH MUSIC PLAYING"
echo "======================================================"
echo ""

echo "⏰ Script will launch EchoPortal in 5 seconds..."
echo "🙏 Make sure to start playing music in Omarchy NOW!"
echo ""
echo "If Omarchy is not playing, the HDMI monitor will likely be SUSPENDED..."
echo ""

for i in {5..1}; do
    echo "Starting in $i seconds... (play music NOW!)"
    sleep 1
done

echo ""
echo "🚀 Launching EchoPortal..."
echo "   Watch for: 'Successfully opened audio device: HDMI monitor'"
echo "   Then watch the debug stats for SampleAvg > 0.000000"
echo ""
echo "   If SampleAvg stays 0.000000, try different device ordering"
echo "   Press ESC to exit EchoPortal"
echo ""

cd ~/Projects/EchoPortal
./echoportal
