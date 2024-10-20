#include "Window.hpp"
#include "Engine.hpp"
#include "imgui.h"
#include "rlImGui.h"
#include <iostream>
#include <chrono>
#include <sys/resource.h>

//TODO: Add a way to change the number of particles
//TODO: Get all UI stuff into a separate file

long get_mem() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  
}

int main() {
    Window window = Window(1600,900);
    window.Init();
    Engine engine = Engine();
    engine.Populate();
    bool paused = true;
    bool diag = false;
    bool showDensity = false;
    auto updatems = std::chrono::milliseconds(0);
    auto simulationms = std::chrono::microseconds(0);
    rlImGuiSetup(true);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(DARKGRAY);
        engine.Draw();

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && showDensity) {
            engine.ShowDensity();
        }

        rlImGuiBegin();
        ImGui::Begin("Settings");
        ImGui::SameLine(ImGui::Checkbox("Show Density", &showDensity));
        ImGui::Checkbox("Diagnostics", &diag);
        ImGui::SliderFloat("Smoothing Radius", &engine.smoothing_radius, 1, 100);
        ImGui::SliderFloat("Threshold", &engine.threshold, 0.1, 1);
        ImGui::SliderFloat("Gravity", &engine.gravity, 0.1, 1);
        ImGui::SliderFloat("Target Density", &engine.targetDensity, 0.1, 7);
        ImGui::SliderFloat("Pressure Multiplier", &engine.pressureMultiplier, 0.0001f, 0.0010f, "%.4f");
        ImGui::ColorEdit3("Particle Color", Particle::color);
        ImGui::SliderFloat("Particle Radius", &Particle::radius, 1, 10);
        ImGui::End();

        if (diag)
        {
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGui::SetNextWindowPos(ImVec2(20, 20));
            ImGui::SetNextWindowSize(ImVec2(250, 150));
            ImGui::Begin("Performance", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
            ImGui::Columns(2, "performance_columns");
            ImGui::Text("Metric"); ImGui::NextColumn();
            ImGui::Text("Value"); ImGui::NextColumn();
            ImGui::Separator();
            ImGui::Text("Update Time"); ImGui::NextColumn();
            ImGui::Text("%lims", updatems.count()); ImGui::NextColumn();
            ImGui::Text("Simulation Time"); ImGui::NextColumn();
            ImGui::Text("%lius", simulationms.count()); ImGui::NextColumn();
            ImGui::Text("Frame Time"); ImGui::NextColumn();
            ImGui::Text("%.fms", GetFrameTime() * 1000); ImGui::NextColumn();
            ImGui::Text("FPS"); ImGui::NextColumn();
            ImGui::Text("%i", GetFPS()); ImGui::NextColumn();
            ImGui::Text("Memory Usage"); ImGui::NextColumn();
            ImGui::Text("%likb",get_mem()); ImGui::NextColumn();
            ImGui::End();
        }
        rlImGuiEnd();
        
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
            engine.SimulationStep();
            end = std::chrono::high_resolution_clock::now();
            simulationms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        }

    }
    CloseWindow();
    return 0;
}

