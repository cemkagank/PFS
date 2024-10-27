#ifndef WINDOW_HPP
#define WINDOW_HPP
#pragma once
#include "raylib.h"

class Window {
    public:
    int WIDTH;
    int HEIGHT;
    Window(int w , int h);
    void Init();

};



#endif // WINDOW_HPPs