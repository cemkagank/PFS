#include "Window.hpp"
#include "Engine.hpp"
#include "imgui.h"
#include "rlImGui.h"
#include <iostream>
#include <chrono>
#include <sys/resource.h>
#include <thread>
#include <raymath.h>
// TODO: Add a way to change the number of particles
// TODO: Get all UI stuff into a separate file

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

    Camera2D cam = {0};
    cam.target = Vector2{800,450};
    cam.offset = Vector2{800,450};
    cam.rotation = 0;
    cam.zoom = 1.0f;

    Vector2 previousMousePosition = {0, 0};
    bool isDragging = false;

    while (!WindowShouldClose())
    {
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            cam.zoom += wheel * 0.05f;  // Adjust 0.05f to change zoom sensitivity
            cam.zoom = Clamp(cam.zoom, 0.3f, 1.0f);  // Limit zoom range
        }
        
        // Add camera panning logic
        if (IsMouseButtonPressed(MOUSE_MIDDLE_BUTTON)) {
            isDragging = true;
            previousMousePosition = GetMousePosition();
        }
        if (IsMouseButtonReleased(MOUSE_MIDDLE_BUTTON)) {
            isDragging = false;
        }
        if (isDragging) {
            Vector2 currentMousePosition = GetMousePosition();
            Vector2 delta = {
                (previousMousePosition.x - currentMousePosition.x) / cam.zoom,
                (previousMousePosition.y - currentMousePosition.y) / cam.zoom
            };
            cam.target = Vector2Add(cam.target, delta);
            previousMousePosition = currentMousePosition;
        }

        BeginDrawing();
        ClearBackground(DARKGRAY);
        
        BeginMode2D(cam);
        engine.Draw();
        EndMode2D();

        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
            engine.Repopulate(GetMousePosition());
        }
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && showDensity) {
            engine.ShowDensity();
        }

        rlImGuiBegin();
        ImGui::Begin("Settings");
        ImGui::SameLine(ImGui::Checkbox("Show Density", &showDensity));
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::Text("Press and hold left mouse button to show density");
            ImGui::EndTooltip();
        }
        ImGui::Checkbox("Diagnostics", &diag);
        ImGui::SliderFloat("Smoothing Radius", &engine.smoothing_radius, 1, 100);
        // ImGui::SliderFloat("Threshold", &engine.threshold, 0.1, 1);
        ImGui::SliderFloat("Gravity", &engine.gravity, 0.1, 1);
        ImGui::SliderFloat("Target Density", &engine.targetDensity, 0.1, 7);
        ImGui::SliderFloat("Camera Zoom", &cam.zoom, 0.3, 1);
        ImGui::SliderFloat("Pressure Multiplier", &engine.pressureMultiplier, 0.0001f, 0.0010f, "%.4f");
        ImGui::ColorEdit3("Particle Color", engine.particle_color);
        ImGui::SliderFloat("Particle Radius", &engine.particle_radius, 1, 10);
        if (ImGui::Button("Pause / Play")) {
            paused = !paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            engine.Reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Populate")) {
        }
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

