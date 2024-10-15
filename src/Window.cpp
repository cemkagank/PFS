#include "Window.hpp"

Window::Window(int w, int h) {
    this->WIDTH = w;
    this->HEIGHT = h;
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_WINDOW_HIGHDPI);
    SetTraceLogLevel(LOG_ERROR);
}

void Window::Init() {
    InitWindow(WIDTH, HEIGHT , "Fluids");
}