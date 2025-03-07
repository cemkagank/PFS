#version 330 core

layout(location = 0) in vec3 vertexPosition;    // Vertex position
layout(location = 1) in vec3 vertexColor;       // Vertex color (if any, can be unused)
layout(location = 2) in mat4 instanceMatrix;    // Instance transformation matrix (passed for each instance)

uniform mat4 mvp;  // Model-view-projection matrix (used to transform the instance)

out vec3 fragColor; // Passing color to the fragment shader

void main() {
    // Apply instance-specific transformation matrix to the vertex position
    vec4 transformedPosition = instanceMatrix * vec4(vertexPosition, 1.0);
    
    // Final position is the result of MVP multiplied by the transformed vertex position
    gl_Position = mvp * transformedPosition;
    
    // Pass the color (can be modified if needed)
    fragColor = vertexColor;
}

