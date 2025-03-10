#version 330

// Input vertex attributes
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;

// Input instance attributes (matrices are stored as 4 vec4 columns)
layout(location = 4) in vec4 instanceTransform0;
layout(location = 5) in vec4 instanceTransform1;
layout(location = 6) in vec4 instanceTransform2;
layout(location = 7) in vec4 instanceTransform3;

// Output vertex attributes
out vec4 fragColor;

// Uniform
uniform mat4 mvp;

void main()
{
    // Construct instance transform matrix
    mat4 instanceMatrix = mat4(
        instanceTransform0,
        instanceTransform1,
        instanceTransform2,
        instanceTransform3
    );

    // Transform vertex position by instance matrix and then by MVP
    vec4 worldPos = instanceMatrix * vec4(vertexPosition, 1.0);
    gl_Position = mvp * worldPos;
    
    // Pass color to fragment shader (using the particle color)
    fragColor = vertexColor;
}
