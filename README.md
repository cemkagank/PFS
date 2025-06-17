# Fluids 

This project is a particle-based computational fluid dynamics (CFD) application, developed as part of a thesis project to explore and analyze state-of-the-art fluid simulation techniques. The application uses C++, raylib for visualization, and ImGui for user interaction, with plans to explore CUDA for enhanced computational performance.


## Features

- **Real-Time Fluid Simulation**: Interactive particle-based fluid simulation, visualized in real time.
- **CUDA Exploration**: Some modules are optimized for GPU performance using CUDA.
- **User Interface with ImGui**: Adjustable simulation parameters through an ImGui-based GUI.
- **Cross-Platform Support**: Built with raylib for easy visualization on various platforms.

## Roadmap

* [X] Complete 2D alpha
* [X] Particle lookup optimizations
* [X] RAII handles
* [X] Transformd to 3D
* [X] Imitate Water Behaviour
* [X] First CUDA kernel
* [X] Add viscocity
* [ ] Grab , Throw forces

## Dependencies
- `raylib` (Must be installed on system DYNAMIC)
- `rlImgui` (Included in project)
- `imgui` (Included in project)
- `CUDA` (Optional, for GPU-based computations)

## Installation

```bash
  git clone https://github.com/cemkagank/PFS
  cd PFS
  mkdir build
  cmake ..
  cd ..
  cmake --build build/
  ./build/bin/fluids
```
## Important

You need te go to raylib.h and change the float3 definitions to whatever you like. For example float3_r. Otherwise you cannot compile the project beacuse of type mismatch between cuda and raylib. 



