#include "Window.hpp"
#include "Engine.hpp"
#include "imgui.h"
#include "rlImGui.h"


int main() {
    Window window = Window(1600,900);
    window.Init();
    Engine engine = Engine();
    engine.Populate();
    bool paused = true;

    rlImGuiSetup(true);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        engine.Draw();

        if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            engine.ShowDensity();
        }

        rlImGuiBegin();
        ImGui::Begin("Settings");
        ImGui::SliderFloat("Smoothing Radius", &engine.smoothing_radius, 1, 100);
        ImGui::SliderFloat("Threshold", &engine.threshold, 0.1, 1);
        ImGui::SliderFloat("Gravity", &engine.gravity, 0.1, 1);
        ImGui::SliderFloat("Target Density", &engine.targetDensity, 0.1, 1);
        ImGui::SliderFloat("Pressure Multiplier", &engine.pressureMultiplier, 0.0001, 0.001);
        ImGui::End();
        rlImGuiEnd();
        
        EndDrawing();
        
        if (IsKeyPressed(KEY_P)) {
            paused = !paused;
        }

        if (!paused) {
            engine.Update();
            engine.SimulationStep();
        }


    }
    CloseWindow();
    return 0;
}