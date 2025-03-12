#include "Window.hpp"
#include "Engine.hpp"
#include "imgui.h"
#include "rlImGui.h"
#include <chrono>
#include <raylib.h>
#include <sys/resource.h>
#include <raymath.h>
#include <rlgl.h>


// TODO: Add a way to change the number of particles
// TODO: Get all UI stuff into a separate file

long get_mem() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  
}

int main() {
    Window window = Window(1920,1080);
    window.Init();
    Engine engine = Engine();
    engine.Populate();
    bool paused = true;
    bool diag = true;
    bool fare = false;
    auto updatems = std::chrono::milliseconds(0);
    auto simulationms = std::chrono::microseconds(0);
    rlImGuiSetup(true);


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

        if (fare) {
            UpdateCamera(&cam,CAMERA_FREE);
        }

        BeginDrawing();
        ClearBackground(GRAY);
        BeginMode3D(cam);
            engine.Draw();
        EndMode3D();
        rlImGuiBegin();
        ImGui::Begin("Settings");
        ImGui::Checkbox("Diagnostics", &diag);
        ImGui::SliderFloat("Smoothing Radius", &engine.smoothing_radius, 1, 100);
        ImGui::SliderFloat("Gravity", &engine.gravity, 0.1, 1);
        ImGui::SliderFloat("Target Density", &engine.targetDensity, 0.1, 7);
        ImGui::SliderFloat("Pressure Multiplier", &engine.pressureMultiplier, 0.000001f, 0.000010f, "%.6f");
        ImGui::ColorEdit3("Particle Color", engine.particle_color);
        ImGui::SliderFloat("Particle Radius", &engine.particle_radius, 0.1, 4);
        if (ImGui::Button("Pause / Play")) {
            paused = !paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            engine.Reset();
        }
       
        if (ImGui::Button("Spawn 100 Particles")) {
            engine.SpawnParticlesAtCenter();
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

