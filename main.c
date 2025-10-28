// EchoPortal - Music-Reactive 3D Portal Visualizer
// Inspired by Asheron's Call (https://www.youtube.com/watch?v=ymiPgYmHjXY)
//
// Dependencies:
//   mesa, glfw-wayland, glew, libpulse, fftw, pkg-config, alsa-lib
//
// Build: make
// Run: ./echoportal

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

// OpenGL and Graphics
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Audio Processing
#include <pulse/simple.h>
#include <fftw3.h>

// Configuration Constants
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define MAX_PARTICLES 2048
#define AUDIO_BUFFER_SIZE 512
#define FFT_SIZE 512
#define SAMPLE_RATE 44100
#define NUM_FREQUENCY_BANDS 3
#define BEAT_THRESHOLD_MULTIPLIER 1.5f

// Audio frequency band definitions (in Hz)
#define LOW_FREQ_MAX 200
#define MID_FREQ_MAX 2000

// Physics and timing constants
#define GRAVITY 0.0f
#define PARTICLE_LIFETIME 3.0f
#define BASE_EMISSION_RATE 50.0f
#define M_PI 3.14159265358979323846f

// GLSL Shader Sources (embedded)
const char* vertex_shader_source = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec4 aColor;\n"
    "layout (location = 2) in float aSize;\n"
    "\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "\n"
    "out vec4 particleColor;\n"
    "out float particleSize;\n"
    "\n"
    "void main() {\n"
    "    gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
    "    particleColor = aColor;\n"
    "    particleSize = aSize;\n"
    "    gl_PointSize = particleSize;\n"
    "}\n";

const char* fragment_shader_source = "#version 330 core\n"
    "in vec4 particleColor;\n"
    "in float particleSize;\n"
    "\n"
    "out vec4 FragColor;\n"
    "\n"
    "void main() {\n"
    "    vec2 coord = gl_PointCoord - vec2(0.5);\n"
    "    float dist = length(coord);\n"
    "    if (dist > 0.5) discard;\n"
    "\n"
    "    float alpha = 1.0 - (dist * 2.0);\n"
    "    alpha = pow(alpha, 2.0);\n"
    "\n"
    "    vec4 color = particleColor;\n"
    "    color.a *= alpha;\n"
    "\n"
    "    FragColor = color;\n"
    "}\n";

const char* vortex_vertex_shader = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "\n"
    "void main() {\n"
    "    gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
    "}\n";

const char* vortex_fragment_shader = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "\n"
    "uniform float time;\n"
    "uniform float audio_energy;\n"
    "\n"
    "float noise(vec2 p) {\n"
    "    return (sin(p.x * 12.9898) + cos(p.y * 78.233)) * 0.5 + 0.5;\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / vec2(1920.0, 1080.0);\n"
    "    uv = uv * 2.0 - 1.0;\n"
    "    uv.x *= 1920.0 / 1080.0;\n"
    "\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float radius = length(uv);\n"
    "\n"
    "    float swirl = sin(angle * 8.0 + time * 2.0 + radius * 4.0) * 0.1;\n"
    "    float inner_radius = 0.3 + swirl * audio_energy;\n"
    "    float outer_radius = 0.7 + swirl * audio_energy;\n"
    "\n"
    "    float mask = smoothstep(inner_radius, outer_radius, radius);\n"
    "    float inner_mask = smoothstep(outer_radius + 0.1, outer_radius, radius);\n"
    "    mask *= (1.0 - inner_mask);\n"
    "\n"
    "    vec3 color = vec3(0.2, 0.4, 0.8); // Blue-purple base\n"
    "    color += vec3(0.1, 0.2, 0.4) * noise(uv * 5.0 + time);\n"
    "    color += vec3(0.8, 0.6, 1.0) * swirl;\n"
    "\n"
    "    FragColor = vec4(color, mask * 0.3);\n"
    "}\n";

// Vector and math utility structs
typedef struct {
    float x, y, z;
} vec3;

typedef struct {
    float r, g, b, a;
} vec4;

typedef struct {
    vec3 pos;
    vec3 vel;
    vec4 color;
    float life;
    float size;
} Particle;

// Global variables
GLFWwindow* window = NULL;
GLuint particleVAO, particleVBO;
GLuint vortexVAO, vortexVBO, vortexEBO;
GLuint particleShader, vortexShader;

fftwf_plan fft_plan;
float* audio_buffer;
float* fft_input;
fftwf_complex* fft_output;
float spectrum[FFT_SIZE/2];
float freq_bands[NUM_FREQUENCY_BANDS];
float beat_energy = 0.0f;
float beat_strength = 0.0f;

Particle particles[MAX_PARTICLES];
int particle_count = 0;

// Audio processing
pa_simple* audio_device = NULL;

// Time tracking
double last_time = 0.0;

// Function prototypes
int init_glfw();
int init_glew();
int init_audio();
int init_shaders();
int init_geometry();
void init_particles();

void process_audio();
void detect_beat();
void update_particles(float dt);
void render();

void cleanup();

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

int main(int argc, char* argv[]) {
    // Initialize everything
    if (!init_glfw()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }

    if (!init_glew()) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return EXIT_FAILURE;
    }

    if (!init_audio()) {
        fprintf(stderr, "Failed to initialize audio\n");
        return EXIT_FAILURE;
    }

    if (!init_shaders()) {
        fprintf(stderr, "Failed to initialize shaders\n");
        return EXIT_FAILURE;
    }

    if (!init_geometry()) {
        fprintf(stderr, "Failed to initialize geometry\n");
        return EXIT_FAILURE;
    }

    init_particles();

    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        float dt = (float)(current_time - last_time);
        last_time = current_time;

        // Process audio if device is available
        if (audio_device) {
            process_audio();
            detect_beat();
        }

        update_particles(dt);
        render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
    return EXIT_SUCCESS;
}

// Implementation of functions
int init_glfw() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 0;
    }

    // Set Wayland hints
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Set fullscreen window hints
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Get the primary monitor for fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    window = glfwCreateWindow(mode->width, mode->height, "EchoPortal", monitor, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 0;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Set input callbacks
    glfwSetKeyCallback(window, key_callback);

    printf("GLFW initialized successfully (%dx%d)\n", mode->width, mode->height);
    return 1;
}

int init_glew() {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return 0;
    }

    printf("GLEW initialized successfully\n");
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    return 1;
}

int create_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Shader compilation failed: %s\n", infoLog);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

int create_shader_program(const char* vertex_source, const char* fragment_source) {
    GLuint vertex_shader = create_shader(vertex_source, GL_VERTEX_SHADER);
    if (!vertex_shader) return 0;

    GLuint fragment_shader = create_shader(fragment_source, GL_FRAGMENT_SHADER);
    if (!fragment_shader) {
        glDeleteShader(vertex_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        fprintf(stderr, "Shader program linking failed: %s\n", infoLog);
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

int init_shaders() {
    particleShader = create_shader_program(vertex_shader_source, fragment_shader_source);
    if (!particleShader) {
        fprintf(stderr, "Failed to create particle shader program\n");
        return 0;
    }

    vortexShader = create_shader_program(vortex_vertex_shader, vortex_fragment_shader);
    if (!vortexShader) {
        fprintf(stderr, "Failed to create vortex shader program\n");
        glDeleteProgram(particleShader);
        return 0;
    }

    printf("Shaders initialized successfully\n");
    return 1;
}

int init_audio() {
    // PulseAudio sample specification
    static const pa_sample_spec ss = {
        .format = PA_SAMPLE_FLOAT32,
        .rate = SAMPLE_RATE,
        .channels = 1  // Mono for visualization
    };

    // Initialize PulseAudio device for monitoring system audio
    // Use NULL for default sink to capture system-wide audio
    audio_device = pa_simple_new(NULL, "EchoPortal", PA_STREAM_RECORD, NULL, "music", &ss, NULL, NULL, NULL);

    if (!audio_device) {
        fprintf(stderr, "Failed to initialize PulseAudio device\n");
        fprintf(stderr, "Make sure PulseAudio is running and audio is available\n");
        return 0;
    }

    // Allocate FFT memory
    audio_buffer = (float*) calloc(AUDIO_BUFFER_SIZE, sizeof(float));
    if (!audio_buffer) {
        fprintf(stderr, "Failed to allocate audio buffer\n");
        return 0;
    }

    fft_input = (float*) fftwf_malloc(sizeof(float) * FFT_SIZE);
    if (!fft_input) {
        fprintf(stderr, "Failed to allocate FFT input buffer\n");
        return 0;
    }

    fft_output = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (FFT_SIZE/2 + 1));
    if (!fft_output) {
        fprintf(stderr, "Failed to allocate FFT output buffer\n");
        return 0;
    }

    // Create FFT plan
    fft_plan = fftwf_plan_dft_r2c_1d(FFT_SIZE, fft_input, fft_output, FFTW_MEASURE);
    if (!fft_plan) {
        fprintf(stderr, "Failed to create FFT plan\n");
        return 0;
    }

    // Seed random for particle spawning
    srand(time(NULL));

    printf("Audio system initialized successfully\n");
    return 1;
}

int init_geometry() {
    // Vertex data for particles (will be updated dynamically)
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * MAX_PARTICLES, NULL, GL_DYNAMIC_DRAW);

    // Position attribute (vec3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, pos));
    glEnableVertexAttribArray(0);

    // Color attribute (vec4)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, color));
    glEnableVertexAttribArray(1);

    // Size attribute (float)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, size));
    glEnableVertexAttribArray(2);

    // Create vortex geometry (simple ring/torus approximation)
#define VORTEX_SEGMENTS 64
    vec3 vortex_vertices[VORTEX_SEGMENTS * 3];
    unsigned int vortex_indices[VORTEX_SEGMENTS * 6];

    const float inner_radius = 0.3f;
    const float outer_radius = 2.0f;
//  const float ring_thickness = 0.1f; // unused

    int vertex_count = 0;
    int index_count = 0;

    for (int i = 0; i < VORTEX_SEGMENTS; i++) {
        float angle = (2.0f * M_PI * i) / VORTEX_SEGMENTS;
        float next_angle = (2.0f * M_PI * (i + 1)) / VORTEX_SEGMENTS;

        // Inner vertices
        vortex_vertices[vertex_count++] = (vec3){inner_radius * cosf(angle), 0.0f, inner_radius * sinf(angle)};
        vortex_vertices[vertex_count++] = (vec3){outer_radius * cosf(angle), 0.0f, outer_radius * sinf(angle)};
        vortex_vertices[vertex_count++] = (vec3){inner_radius * cosf(next_angle), 0.0f, inner_radius * sinf(next_angle)};

        vortex_indices[index_count++] = (i * 3);
        vortex_indices[index_count++] = (i * 3 + 1);
        vortex_indices[index_count++] = (i * 3 + 2);
    }

    glGenVertexArrays(1, &vortexVAO);
    glGenBuffers(1, &vortexVBO);
    glGenBuffers(1, &vortexEBO);

    glBindVertexArray(vortexVAO);
    glBindBuffer(GL_ARRAY_BUFFER, vortexVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vortex_vertices), vortex_vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vortexEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vortex_indices), vortex_indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
    glEnableVertexAttribArray(0);

    printf("Geometry initialized successfully\n");
    return 1;
}

void init_particles() {
    // Initialize particle array
    memset(particles, 0, sizeof(particles));
}

void process_audio() {
    if (!audio_device) return;

    // Read audio samples from PulseAudio
    int pa_result = pa_simple_read(audio_device, audio_buffer,
                                 AUDIO_BUFFER_SIZE * sizeof(float), NULL);
    if (pa_result < 0) {
        fprintf(stderr, "Failed to read audio data (error %d)\n", pa_result);
        // Could disable audio processing if we get persistent errors
        return;
    }

    // Copy to FFT input buffer
    memcpy(fft_input, audio_buffer, AUDIO_BUFFER_SIZE * sizeof(float));

    // Zero-pad remaining FFT input (if buffer is smaller than FFT_SIZE)
    if (FFT_SIZE > AUDIO_BUFFER_SIZE) {
        memset(fft_input + AUDIO_BUFFER_SIZE, 0,
               (FFT_SIZE - AUDIO_BUFFER_SIZE) * sizeof(float));
    }

    // Execute FFT
    fftwf_execute(fft_plan);

    // Convert complex FFT output to magnitude spectrum
    for (int i = 0; i < FFT_SIZE/2; i++) {
        spectrum[i] = sqrtf(fft_output[i][0] * fft_output[i][0] +
                           fft_output[i][1] * fft_output[i][1]);
    }

    // Calculate frequency bands
    // Low freq: 0-200Hz, Mid freq: 200-2000Hz, High freq: 2000-22050Hz
    const int nyquist = SAMPLE_RATE / 2;
    const int low_bins = (int)(LOW_FREQ_MAX * (FFT_SIZE / 2.0f) / nyquist);
    const int mid_bins = (int)(MID_FREQ_MAX * (FFT_SIZE / 2.0f) / nyquist);

    float low_energy = 0.0f, mid_energy = 0.0f, high_energy = 0.0f;
    int low_count = 0, mid_count = 0, high_count = 0;

    for (int i = 1; i < FFT_SIZE/2; i++) {  // Start from 1 to skip DC component
        if (i <= low_bins) {
            low_energy += spectrum[i];
            low_count++;
        } else if (i <= mid_bins) {
            mid_energy += spectrum[i];
            mid_count++;
        } else {
            high_energy += spectrum[i];
            high_count++;
        }
    }

    // Normalize by bin count and convert to 0-1 range
    if (low_count > 0) freq_bands[0] = low_energy / (low_count * 100.0f);
    if (mid_count > 0) freq_bands[1] = mid_energy / (mid_count * 50.0f);
    if (high_count > 0) freq_bands[2] = high_energy / (high_count * 25.0f);

    // Clamp to reasonable range
    for (int i = 0; i < NUM_FREQUENCY_BANDS; i++) {
        if (freq_bands[i] > 1.0f) freq_bands[i] = 1.0f;
    }
}

void detect_beat() {
    if (!audio_device) return;

    // Calculate RMS energy of low frequencies (bass-heavy detection)
    float current_energy = 0.0f;
    for (int i = 1; i < AUDIO_BUFFER_SIZE; i++) {
        current_energy += audio_buffer[i] * audio_buffer[i];
    }
    current_energy = sqrtf(current_energy / AUDIO_BUFFER_SIZE);

    // Simple low-pass filter for energy envelope
    static float smoothed_energy = 0.0f;
    smoothed_energy = 0.9f * smoothed_energy + 0.1f * current_energy;

    // Adaptive threshold based on recent average energy
    static float energy_history[100] = {0};
    static int history_index = 0;
    static float energy_sum = 0.0f;

    energy_sum -= energy_history[history_index];
    energy_history[history_index] = current_energy;
    energy_sum += current_energy;
    history_index = (history_index + 1) % 100;

    float avg_energy = energy_sum / 100.0f;
    float threshold = avg_energy * BEAT_THRESHOLD_MULTIPLIER;

    // Beat detection: current energy significantly above threshold
    static int beat_cooldown = 0;
    if (current_energy > threshold && smoothed_energy > threshold && beat_cooldown <= 0) {
        beat_strength = (current_energy - threshold) / (avg_energy * 2.0f); // Normalize
        if (beat_strength > 1.0f) beat_strength = 1.0f;

        beat_cooldown = SAMPLE_RATE / 180; // ~180 BPM minimum interval
    } else {
        beat_strength *= 0.95f; // Decay

        if (beat_cooldown > 0) beat_cooldown--;
    }

    // Store for use by rendering systems
    beat_energy = current_energy;
}

// Simple noise function for organic particle movement
float simple_noise(float x, float y, float time) {
    return sinf(x * 12.9898f + time) * cosf(y * 78.233f - time * 0.5f);
}

// Clamp function for values
float clamp_float(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void update_particles(float dt) {
    // Calculate emission rate based on audio data
    float base_emission = BASE_EMISSION_RATE;
    float beat_multiplier = 1.0f + beat_strength * 10.0f; // Up to 10x on strong beats
    float audio_multiplier = 1.0f + (freq_bands[2] * 2.0f); // High freq increases density
    int emission_rate = (int)(base_emission * beat_multiplier * audio_multiplier);

    // Spawn new particles
    static float spawn_timer = 0.0f;
    spawn_timer += dt;

    int particles_to_spawn = (int)(spawn_timer * emission_rate);
    if (particles_to_spawn > 0 || particle_count == 0) {
        for (int i = 0; i < particles_to_spawn && particle_count < MAX_PARTICLES; i++) {
            // Find dead particle
            int particle_index = -1;
            for (int j = 0; j < MAX_PARTICLES; j++) {
                if (particles[j].life <= 0.0f) {
                    particle_index = j;
                    break;
                }
            }

            if (particle_index == -1) break; // No available particles

            Particle* p = &particles[particle_index];

            // Spawn at vortex rim with random angle
            float spawn_angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;

            // Wide vortex radius modulated by low frequencies
            float vortex_radius = 2.0f + freq_bands[0] * 0.5f; // Wider with bass

            p->pos.x = cosf(spawn_angle) * vortex_radius;
            p->pos.y = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // Small y variation
            p->pos.z = sinf(spawn_angle) * vortex_radius;

            // Initial velocity inward, accelerated by beat
            float speed = 0.5f + beat_strength * 2.0f;
            p->vel.x = -cosf(spawn_angle) * speed;
            p->vel.y = 0.0f;
            p->vel.z = -sinf(spawn_angle) * speed;

            // Base blue-purple color
            p->color.r = 0.2f;
            p->color.g = 0.4f;
            p->color.b = 0.8f;
            p->color.a = 0.8f;

            p->size = 3.0f + ((float)rand() / RAND_MAX) * 2.0f;
            p->life = PARTICLE_LIFETIME;

            particle_count++;
        }
        spawn_timer = 0.0f;
    }

    // Update existing particles
    for (int i = 0; i < MAX_PARTICLES; i++) {
        Particle* p = &particles[i];
        if (p->life <= 0.0f) continue;

        // Update physics
        p->vel.y += GRAVITY * dt; // Gravity effect

        // Add noise perturbations based on mid frequencies (organic twisting)
        float noise_scale = freq_bands[1] * 5.0f; // Mid freq causes irregular motion
        float noise_x = simple_noise(p->pos.x * 0.1f, p->pos.z * 0.1f, last_time) * noise_scale;
        float noise_z = simple_noise(p->pos.z * 0.1f, p->pos.y * 0.1f, last_time) * noise_scale;

        p->vel.x += noise_x * dt;
        p->vel.z += noise_z * dt;

        // Apply velocity
        p->pos.x += p->vel.x * dt;
        p->pos.y += p->vel.y * dt;
        p->pos.z += p->vel.z * dt;

        // Spiral inward motion (parametric helix)
        float distance_from_center = sqrtf(p->pos.x * p->pos.x + p->pos.z * p->pos.z);
        if (distance_from_center > 0.1f) {
            float angle = atan2f(p->pos.z, p->pos.x);
            float target_radius = distance_from_center * 0.99f; // Gradually spiral inward

            // Add spiral rotation
            float spiral_speed = 1.0f + beat_strength; // Speed up on beats
            angle += spiral_speed * dt;

            // Apply mid frequency irregularity to rotation
            angle += simple_noise(angle, distance_from_center, last_time * 0.5f)
                   * freq_bands[1] * 2.0f;

            p->pos.x = cosf(angle) * target_radius;
            p->pos.z = sinf(angle) * target_radius;
        }

        // Color modulation based on frequency bands
        // Low freq: Enhance blue/purple saturation
        p->color.r = clamp_float(p->color.r - freq_bands[0] * 0.1f, 0.1f, 1.0f);
        p->color.b = clamp_float(p->color.b + freq_bands[0] * 0.3f, 0.1f, 1.0f);

        // Mid freq: Add green/teal accents
        p->color.g = clamp_float(p->color.g + freq_bands[1] * 0.4f, 0.1f, 1.0f);

        // High freq: White/gold sparks
        if (freq_bands[2] > 0.3f) {
            p->color.r = clamp_float(p->color.r + freq_bands[2] * 0.5f, 0.1f, 1.0f);
            p->color.g = clamp_float(p->color.g + freq_bands[2] * 0.3f, 0.1f, 1.0f);
            p->color.b = clamp_float(p->color.b + freq_bands[2] * 0.2f, 0.1f, 1.0f);
        }

        // Decay particles near center (fade to whiteness)
        float center_distance = sqrtf(p->pos.x * p->pos.x + p->pos.z * p->pos.z);
        if (center_distance < 0.3f) {
            float fade_factor = center_distance / 0.3f;
            p->color.r = p->color.r * fade_factor + 1.0f * (1.0f - fade_factor);
            p->color.g = p->color.g * fade_factor + 1.0f * (1.0f - fade_factor);
            p->color.b = p->color.b * fade_factor + 1.0f * (1.0f - fade_factor);
            p->size *= 0.8f; // Shrink at center
        }

        // Update alpha based on life
        p->color.a = (p->life / PARTICLE_LIFETIME) * 0.8f;

        // Age and possibly kill particle
        p->life -= dt;

        // Kill particles that reached center or expired
        if (center_distance < 0.05f || p->life <= 0.0f) {
            p->life = -1.0f;
            particle_count--;
        }
    }
}

// Matrix multiplication helper functions
void mat4_identity(float* m) {
    for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

void mat4_perspective(float fovy, float aspect, float zNear, float zFar, float* result) {
    float f = 1.0f / tanf(fovy * 0.5f);

    result[0] = f / aspect; result[4] = 0.0f; result[8]  = 0.0f;                        result[12] = 0.0f;
    result[1] = 0.0f;       result[5] = f;    result[9]  = 0.0f;                        result[13] = 0.0f;
    result[2] = 0.0f;       result[6] = 0.0f; result[10] = (zFar + zNear)/(zNear - zFar); result[14] = (2.0f * zFar * zNear)/(zNear - zFar);
    result[3] = 0.0f;       result[7] = 0.0f; result[11] = -1.0f;                       result[15] = 0.0f;
}

void mat4_translate(float x, float y, float z, float* result) {
    mat4_identity(result);
    result[12] = x; result[13] = y; result[14] = z;
}

void mat4_rotate_y(float angle, float* result) {
    float c = cosf(angle);
    float s = sinf(angle);

    result[0] = c;    result[4] = 0.0f; result[8]  = s;    result[12] = 0.0f;
    result[1] = 0.0f; result[5] = 1.0f; result[9]  = 0.0f; result[13] = 0.0f;
    result[2] = -s;   result[6] = 0.0f; result[10] = c;    result[14] = 0.0f;
    result[3] = 0.0f; result[7] = 0.0f; result[11] = 0.0f; result[15] = 1.0f;
}

void mat4_look_at(float eyex, float eyey, float eyez,
                  float centerx, float centery, float centerz,
                  float upx, float upy, float upz, float* result) {
    // Simplified look-at for orbital camera
    float forwardx = centerx - eyex;
    float forwardy = centery - eyey;
    float forwardz = centerz - eyez;

    float len = sqrtf(forwardx*forwardx + forwardy*forwardy + forwardz*forwardz);
    forwardx /= len; forwardy /= len; forwardz /= len;

    float rightx = forwardy*upz - forwardz*upy;
    float righty = forwardz*upx - forwardx*upz;
    float rightz = forwardx*upy - forwardy*upx;
    len = sqrtf(rightx*rightx + righty*righty + rightz*rightz);
    rightx /= len; righty /= len; rightz /= len;

    float upvecx = righty*forwardz - rightz*forwardy;
    float upvecy = rightz*forwardx - rightx*forwardz;
    float upvecz = rightx*forwardy - righty*forwardx;

    result[0] = rightx;    result[4] = upvecx;    result[8]  = -forwardx; result[12] = -(rightx*eyex + righty*eyey + rightz*eyez);
    result[1] = righty;    result[5] = upvecy;    result[9]  = -forwardy; result[13] = -(upvecx*eyey + upvecy*eyey + upvecz*eyez);
    result[2] = rightz;    result[6] = upvecz;    result[10] = -forwardz; result[14] = -(upvecx*eyez + upvecy*eyez + upvecz*eyez);
    result[3] = 0.0f;      result[7] = 0.0f;      result[11] = 0.0f;      result[15] = 1.0f;
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Get window dimensions for aspect ratio
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    float aspect = (float)width / (float)height;

    // Setup matrices
    float projection[16];
    mat4_perspective(M_PI * 0.5f, aspect, 0.1f, 10.0f, projection);

    float view[16];
    // Orbital camera around portal
    float camera_angle = (float)last_time * 0.2f; // Slow orbital motion
    float camera_distance = 4.0f + beat_strength * 0.5f; // Pull closer on beats
    float eyex = sinf(camera_angle) * camera_distance;
    float eyez = cosf(camera_angle) * camera_distance;
    float eyey = 0.5f;

    mat4_look_at(eyex, eyey, eyez, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, view);

    float model[16];
    mat4_identity(model);

    // Render vortex background first
    glDisable(GL_DEPTH_TEST); // No depth for background
    glUseProgram(vortexShader);

    int timeLoc = glGetUniformLocation(vortexShader, "time");
    glUniform1f(timeLoc, (float)last_time);

    int energyLoc = glGetUniformLocation(vortexShader, "audio_energy");
    glUniform1f(energyLoc, beat_energy);

    glBindVertexArray(vortexVAO);
    glDrawElements(GL_TRIANGLES, VORTEX_SEGMENTS * 6, GL_UNSIGNED_INT, 0);

    // Enable depth testing for particles
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    // Render particles
    glUseProgram(particleShader);

    int modelLoc = glGetUniformLocation(particleShader, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model);

    int viewLoc = glGetUniformLocation(particleShader, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);

    int projLoc = glGetUniformLocation(particleShader, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);

    // Update particle VBO with current data
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(particles), particles);

    // Draw particles
    glBindVertexArray(particleVAO);
    if (particle_count > 0) {
        glDrawArrays(GL_POINTS, 0, particle_count);
    }

    // Reset state
    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glUseProgram(0);
}

void cleanup() {
    // Clean up audio resources
    if (audio_device) {
        pa_simple_free(audio_device);
        audio_device = NULL;
    }

    if (audio_buffer) {
        free(audio_buffer);
        audio_buffer = NULL;
    }

    if (fft_input) {
        fftwf_free(fft_input);
        fft_input = NULL;
    }

    if (fft_output) {
        fftwf_free(fft_output);
        fft_output = NULL;
    }

    if (fft_plan) {
        fftwf_destroy_plan(fft_plan);
        fftwf_cleanup();
    }

    // Clean up OpenGL resources
    if (particleShader) glDeleteProgram(particleShader);
    if (vortexShader) glDeleteProgram(vortexShader);

    if (particleVAO) glDeleteVertexArrays(1, &particleVAO);
    if (particleVBO) glDeleteBuffers(1, &particleVBO);

    if (vortexVAO) glDeleteVertexArrays(1, &vortexVAO);
    if (vortexVBO) glDeleteBuffers(1, &vortexVBO);
    if (vortexEBO) glDeleteBuffers(1, &vortexEBO);

    // Clean up GLFW
    glfwTerminate();

    printf("Cleanup completed successfully\n");
}
