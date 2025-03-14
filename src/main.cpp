#include "Window.hpp"
#include "Engine.hpp"
#include "imgui.h"
#include "rlImGui.h"
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <chrono>
#include "UI.hpp"


int main() {
    Window window = Window(1920,1080);
    window.Init();
    Engine engine = Engine();
    engine.Populate();
    bool paused = true;
    bool fare = false;
    bool ui_enabled =true;
    auto updatems = std::chrono::milliseconds(0);
    auto simulationms = std::chrono::microseconds(0);
    rlImGuiSetup(true);

    UI ui = UI(engine, updatems, simulationms);


    Camera3D cam;
    cam.target = {0,0,0};
    cam.position = {20,20,30};
    cam.up = {0,1,0};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    rlEnableDepthTest();


    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_TAB))
            fare = !fare;
        if(IsKeyPressed(KEY_LEFT_SHIFT))
            ui_enabled = !ui_enabled;
        if (fare) {
            UpdateCamera(&cam,CAMERA_FREE);
        }



        BeginDrawing();
        ClearBackground(GRAY);
        BeginMode3D(cam);
            engine.Draw();
        EndMode3D();

        if (ui_enabled)
            ui.Render();
        
        EndDrawing();
        if (IsKeyPressed(KEY_P)) {
            paused = !paused;
        }
        
        if (!paused) {
            auto start = std::chrono::high_resolution_clock::now();
            engine.Update();
            auto end = std::chrono::high_resolution_clock::now();
            updatems = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            start = std::chrono::high_resolution_clock::now();
           
            end = std::chrono::high_resolution_clock::now();
            simulationms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        }

    }
    CloseWindow();
    return 0;
}

