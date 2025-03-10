#version 330

// Input vertex attributes
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;

// Output vertex attributes
out vec4 fragColor;

// Uniform
uniform mat4 mvp;
uniform mat4 matModel;

void main()
{
    // Transform vertex position
    gl_Position = mvp * vec4(vertexPosition, 1.0);
    
    // Pass color to fragment shader
    fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Force red color for testing
}

