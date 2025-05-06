#include "Window.hpp"
#include <raylib.h>

Window::Window(int w, int h) {
    this->WIDTH = w;
    this->HEIGHT = h;
    SetConfigFlags( FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_FULLSCREEN_MODE);
}

void Window::Init() {
    InitWindow(WIDTH, HEIGHT , "Fluids");
}
