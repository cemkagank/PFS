#version 330

// Per-vertex attributes
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;

// Per-instance transform matrix (occupies attribute locations 4-7)
layout(location = 4) in mat4 instanceTransform;

uniform mat4 mvp;

out vec4 fragColor;

void main()
{
    gl_Position = mvp * instanceTransform * vec4(vertexPosition, 1.0);
    fragColor = vertexColor;
}
