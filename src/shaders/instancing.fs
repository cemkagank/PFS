#version 330 core

in vec3 fragColor;  // Color from the vertex shader

out vec4 finalColor;

void main() {
    // Set the fragment color
    finalColor = vec4(fragColor, 1.0);
}

