#include "Window.hpp"

Window::Window(int w, int h) {
    this->WIDTH = w;
    this->HEIGHT = h;
    SetConfigFlags( FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT);
    SetTargetFPS(60);
    SetTraceLogLevel(LOG_ERROR);
}

void Window::Init() {
    InitWindow(WIDTH, HEIGHT , "Fluids");
}