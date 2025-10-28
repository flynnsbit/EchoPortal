# EchoPortal - Music-Reactive 3D Portal Visualizer

A mesmerizing real-time music visualizer inspired by Asheron's Call, that transforms your music into a swirling vortex of particles flowing through a mysterious portal in deep space.

## 🎵 Features

- **Real-time Audio Analysis**: Captures system audio via PulseAudio and analyzes it using FFT
- **Interactive Particle System**: Up to 2048 swirling particles that react to music
- **Adaptive Beat Detection**: Intelligent rhythm analysis with adaptive thresholds
- **Frequency Band Mapping**: Separate visual response for bass, mid, and high frequencies
- **3D Orbital Camera**: Smooth orbiting camera with beat-responsive distance
- **Procedural Vortex**: Animated background effects reacting to audio energy
- **Full Wayland Support**: Native support for Wayland compositors (Hyprland)
- **Performance Optimized**: Targets 60 FPS with low CPU/GPU usage

## 🎨 Visual Response

### Beat Response
- **Intensified emission**: Dramatic particle rate increase on strong beats
- **Accelerated flow**: Particles speed up when beats are detected
- **Camera pull**: View comes closer to the portal center on beats
- **Pulsed brightness**: Energy pulses through the vortex system

### Frequency Response
- **Low/Bass (0-200Hz)**: Blue-purple color enhancements, vortex expansion
- **Mid/Vocals (200-2000Hz)**: Green-teal accents, irregular swirling motion
- **High/Treble (2000-22050Hz)**: Sparkling white/gold particles, increased density

## 🚀 Quick Start

### Prerequisites
Make sure these packages are installed (Arch Linux):
```bash
sudo pacman -Syu mesa glfw-wayland glew libpulse fftw pkg-config alsa-lib
```

### Build
```bash
make          # Compile the application
make install-deps  # Install dependencies if needed
```

### Run
```bash
./echoportal  # Run fullscreen
```

### Controls
- **ESC**: Exit the application
- **Music playing elsewhere**: EchoPortal will automatically capture and visualize your system's audio

## 🔧 System Requirements

- **OS**: Linux (optimized for Arch Linux)
- **Graphics**: OpenGL 3.3 Core capable GPU
- **Audio**: PulseAudio running (default on most Linux distributions)
- **Display**: Wayland compositor (Hyprland recommended)
- **Memory**: ~50MB RAM
- **CPU/GPU**: Moderate hardware (targets <20% usage on typical systems)

## 🏗️ Architecture

### Core Components
- **Audio Processing**: PulseAudio capture → FFTW analysis → frequency band extraction
- **Beat Detection**: Adaptive thresholding with rolling energy averages
- **Particle System**: 1000-5000 particles with lifecycle management
- **Rendering Pipeline**: OpenGL 3.3 with instanced drawing and matrix transformations
- **3D Camera**: Orbital motion around portal with audio-reactive adjustments

### Audio Data Flow
```
System Audio → PulseAudio Capture → FFT Analysis → Frequency Bands → Beat Detection
                                                                ↓
                                              Particle Generation & Color Modulation
                                                                ↓
                                   3D Rendering with Orbital Camera & Vortex Effects
```

## 🎯 Technical Implementation

### Audio Processing
- **Sample Rate**: 44.1kHz mono
- **FFT Size**: 512 bins (0.0116s window)
- **Band Separation**: Low/Mid/High frequency isolation
- **Beat Algorithm**: Rolling energy average with adaptive thresholds

### Graphics Pipeline
- **Shaders**: Embedded GLSL for particles and vortex effects
- **Blending**: Additive particle rendering for glow effects
- **Geometry**: Procedural vortex with time-varying swirls
- **Matrices**: Custom 4x4 matrix math for 3D transformations

### Performance Optimizations
- **Circular Particle Management**: Recycling dead particles eliminates allocation overhead
- **VBO Dynamic Updates**: Efficient GPU particle data updates
- **Adaptive Emission**: Audio-responsive particle rates prevent overload
- **VSYNC Enabled**: Consistent 60 FPS target with GPU synchronization

## 🎊 Release Notes

This is a fully functional implementation of the EchoPortal music visualizer with all specified features:

- ✅ Complete audio capture and analysis system
- ✅ Advanced beat detection and frequency band processing
- ✅ Comprehensive particle physics with audio reactivity
- ✅ 3D orbital rendering with procedural vortex background
- ✅ Full Wayland/Windows support and proper cleanup
- ✅ Performance optimized for demanding audio-reactive graphics

## 🎮 Usage Tips

- **Maximum Impact**: Play music with strong bass and clear beats for most spectacular visuals
- **Genre Recommendations**: Electronic, EDM, hip-hop, and bass-heavy tracks work best
- **System Audio**: Ensure your music player outputs audio through your system's default PulseAudio device
- **Monitor Setup**: Best experienced on high-refresh-rate displays in fullscreen

## 🔧 Troubleshooting

### Audio Issues with Specific Players

**Omarchy (Wayland Music Player):**
- **Issue**: EchoPortal may not capture audio from Omarchy
- **Cause**: Okarchy's different audio routing (uses PipeWire/PulseAudio sinks)
- **Solution**: EchoPortal automatically tries multiple audio sources - it should detect and work
- **Verification**: Look for "Audio system initialized successfully using..." in console

**Other Players:**
If music reactivity doesn't work:
1. Start playing music first
2. Check console output - it shows which audio device is active
3. Try restarting EchoPortal after starting music

### Visual / Performance Issues

**Low FPS or sluggish performance:**
- Check that OpenGL 3.3 is supported: `glxinfo | grep "OpenGL"`
- Verify hardware acceleration: `nvidia-smi` (if using NVIDIA)

**Black screen (before debugging fixes):**
- This was a bug in v1.0 - make sure you're using the latest version
- The current version shows a purple background with test particles immediately

**Console output explained:**
```bash
DEBUG: Frame 120 | Particles: 45 | Beat: 0.234 | Freq: 0.123 0.456 0.789 | Audio: 1
        ^ Frame #    ^ Alive part  ^ Beat str  ^ Bass/Mid/High ^ Audio work
```

## 🛠️ Development

### Project Structure
```
EchoPortal/
├── main.c          # Monolithic application (~1600 LOC)
├── Makefile        # Build automation
├── README.md       # This file
└── .gitignore      # Git exclusions (if added)
```

### Build System
- **Compiler**: GCC with C99 standard
- **Optimization**: -O2 for performance
- **Libraries**: GLFW-GL3, GLEW, OpenGL, PulseAudio, FFTW
- **Warnings**: All warnings enabled for code quality

### Future Enhancements
- Additional shader effects for more visual variety
- Multiple vortex configurations
- Customizable color schemes
- Audio input device selection
- Screenshot/movie recording features

### Debug Mode
Run with verbose console output for troubleshooting:
```bash
./echoportal  # Automatically shows debug info every second
# Look for: Audio device detection, particle counts, frequency values
```

---

*EchoPortal - Bringing music to life through mesmerizing 3D visualization* ✨
