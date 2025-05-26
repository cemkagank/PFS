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
    
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        engine.Reset();
    }

    // Draw a line here to separate the buttons
    ImGui::Separator();
    if (ImGui::Button("Populate : 10^3")) {
        engine.Reset();
        engine.Populate(10);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 15^3")) {
        engine.Reset();
        engine.Populate(15);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 20^3")) {
        engine.Reset();
        engine.Populate(20);
    }


    if (ImGui::Button("Populate : 24^3")) {
        engine.Reset();
        engine.Populate(24);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 28^3")) {
        engine.Reset();
        engine.Populate(28);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 32^3")) {
        engine.Reset();
        engine.Populate(32);
    }
    if (ImGui::Button("Populate : 34^3")) {
        engine.Reset();
        engine.Populate(34);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 36^3")) {
        engine.Reset();
        engine.Populate(36);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 38^3")) {
        engine.Reset();
        engine.Populate(38);
    }
    if (ImGui::Button("Populate : 42^3")) {
        engine.Reset();
        engine.Populate(42);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 44^3")) {
        engine.Reset();
        engine.Populate(44);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 48^3")) {
        engine.Reset();
        engine.Populate(48);
    }
    if (ImGui::Button("Populate : 52^3")) {
        engine.Reset();
        engine.Populate(52);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 56^3")) {
        engine.Reset();
        engine.Populate(56);
    }
    ImGui::SameLine();
    if (ImGui::Button("Populate : 64^3")) {
        engine.Reset();
        engine.Populate(64);
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
        ImGui::Text("Simulation Time"); ImGui::NextColumn();
        ImGui::Text("%lius", simms.count()); ImGui::NextColumn();
        ImGui::Text("Frame Time"); ImGui::NextColumn();
        ImGui::Text("%.fms", GetFrameTime() * 1000); ImGui::NextColumn();
        ImGui::Text("FPS"); ImGui::NextColumn();
        ImGui::Text("%i", GetFPS()); ImGui::NextColumn();
    
        ImGui::End();
    }
    rlImGuiEnd();


}
