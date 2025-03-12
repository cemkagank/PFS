#ifndef UI_HPP
#define UI_HPP
#include "Engine.hpp"
#include <sys/resource.h>
#include <chrono>


class UI {
    private:
    
    Engine &engine;
    bool diag = true;
    long get_mem();
    std::chrono::milliseconds &updatems;
    std::chrono::microseconds &simms;


    public:     
    UI(Engine &engine, std::chrono::milliseconds &up, std::chrono::microseconds &sim);
    void Render();
    void Toggle();
    void Extend();
    
};










#endif //UI_HPPs
