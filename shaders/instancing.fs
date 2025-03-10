#version 330

// Input vertex attributes (from vertex shader)
in vec4 fragColor;

// Output fragment color
out vec4 finalColor;

void main()
{
    finalColor = fragColor;
}
