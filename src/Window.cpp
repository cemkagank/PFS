#include "Window.hpp"
#include <raylib.h>

Window::Window(int w, int h) {
    this->WIDTH = w;
    this->HEIGHT = h;
    // FLAG_WINDOW_HIGHDPI enables Retina support on macOS
    // This allows proper high-DPI rendering on Retina displays
    // FLAG_FULLSCREEN_MODE removed - use windowed mode by default
    SetConfigFlags(FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    SetTraceLogLevel(LOG_ERROR);
}

void Window::Init() {
    InitWindow(WIDTH, HEIGHT, "Fluids");
    // Enable window resizing on macOS
    SetWindowState(FLAG_WINDOW_RESIZABLE);
}
