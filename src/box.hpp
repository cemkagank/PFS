#ifndef BOX_HPP
#define BOX_HPP

#include <raylib.h>
#include <raymath.h>

struct box {
    Vector3 max;
    Vector3 min;
    
    void Draw() {
        // Draw the box as wireframe
        DrawLine3D(Vector3{min.x, min.y, min.z}, Vector3{max.x, min.y, min.z}, WHITE);
        DrawLine3D(Vector3{min.x, min.y, min.z}, Vector3{min.x, max.y, min.z}, WHITE);
        DrawLine3D(Vector3{min.x, min.y, min.z}, Vector3{min.x, min.y, max.z}, WHITE);
        
        DrawLine3D(Vector3{max.x, min.y, min.z}, Vector3{max.x, max.y, min.z}, WHITE);
        DrawLine3D(Vector3{max.x, min.y, min.z}, Vector3{max.x, min.y, max.z}, WHITE);
        
        DrawLine3D(Vector3{min.x, max.y, min.z}, Vector3{max.x, max.y, min.z}, WHITE);
        DrawLine3D(Vector3{min.x, max.y, min.z}, Vector3{min.x, max.y, max.z}, WHITE);
        
        DrawLine3D(Vector3{min.x, min.y, max.z}, Vector3{max.x, min.y, max.z}, WHITE);
        DrawLine3D(Vector3{min.x, min.y, max.z}, Vector3{min.x, max.y, max.z}, WHITE);
        
        DrawLine3D(Vector3{max.x, max.y, min.z}, Vector3{max.x, max.y, max.z}, WHITE);
        DrawLine3D(Vector3{max.x, min.y, max.z}, Vector3{max.x, max.y, max.z}, WHITE);
        DrawLine3D(Vector3{min.x, max.y, max.z}, Vector3{max.x, max.y, max.z}, WHITE);
    }
};

#endif // BOX_HPP 