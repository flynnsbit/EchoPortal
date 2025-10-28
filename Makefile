# EchoPortal Makefile

CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra
LDFLAGS = -lglfw -lGLEW -lGL -lpulse-simple -lfftw3f -lm
TARGET = echoportal
SOURCES = main.c

.PHONY: all clean install-deps run

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install-deps:
	sudo pacman -Syu mesa glfw-wayland glew libpulse fftw pkg-config alsa-lib

run: $(TARGET)
	./$(TARGET)

debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all         - Build the application (default)"
	@echo "  clean       - Remove built files"
	@echo "  install-deps- Install required dependencies (requires sudo)"
	@echo "  run         - Build and run the application"
	@echo "  debug       - Build with debug symbols"
	@echo "  help        - Show this help message"
