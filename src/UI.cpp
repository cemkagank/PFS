#include "UI.hpp"
#include "imgui.h"
#include "rlImGui.h"

UI::UI(Engine &engine, std::chrono::milliseconds &up, std::chrono::microseconds &sim) : 
    engine(engine),
    updatems(up), 
    simms(sim)
{
}


long UI::get_mem() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  
}

void UI::Render() {
    rlImGuiBegin();
    ImGui::Begin("Settings");
    ImGui::Checkbox("Diagnostics", &diag);
    ImGui::SliderFloat("Smoothing Radius", &engine.smoothing_radius, 1, 100);
    ImGui::SliderFloat("Gravity", &engine.gravity, 0.1, 1);
    ImGui::SliderFloat("Target Density", &engine.targetDensity, 0.1, 7);
    ImGui::SliderFloat("Pressure Multiplier", &engine.pressureMultiplier, 0.000001f, 0.000010f, "%.6f");
    ImGui::ColorEdit3("Particle Color", engine.particle_color);
    ImGui::SliderFloat("Particle Radius", &engine.particle_radius, 0.1, 4);
   
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        engine.Reset();
    }

    if (ImGui::Button("Spawn 1000 Particles")) {
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
        ImGui::Text("%lius", simms.count()); ImGui::NextColumn();
        ImGui::Text("Frame Time"); ImGui::NextColumn();
        ImGui::Text("%.fms", GetFrameTime() * 1000); ImGui::NextColumn();
        ImGui::Text("FPS"); ImGui::NextColumn();
        ImGui::Text("%i", GetFPS()); ImGui::NextColumn();
        ImGui::Text("Memory Usage"); ImGui::NextColumn();
        ImGui::Text("%likb",get_mem()); ImGui::NextColumn();
        ImGui::End();
    }
    rlImGuiEnd();


}
