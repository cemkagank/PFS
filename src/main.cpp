#include "Window.hpp"
#include "Engine.hpp"

int main() {
    Window window = Window(1600,900);
    window.Init();
    Engine engine = Engine();
    engine.Populate();
    bool paused = true;
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        engine.Draw();

        if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            engine.ShowDensity();
        }
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