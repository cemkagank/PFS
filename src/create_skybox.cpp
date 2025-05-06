#include <raylib.h>
#include <stdio.h>

int main() {
    // Initialize window
    InitWindow(1, 1, "Skybox Generator");
    
    // Create six 512x512 images for each face of the cubemap
    Image faces[6];
    for (int i = 0; i < 6; i++) {
        faces[i] = GenImageColor(512, 512, BLANK);
    }
    
    // Fill each face with a gradient
    for (int face = 0; face < 6; face++) {
        for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
                // Calculate normalized coordinates
                float nx = (float)x / 512.0f;
                float ny = (float)y / 512.0f;
                
                // Create different gradients for each face
                Color color;
                switch (face) {
                    case 0: // Right face
                        color = {
                            (unsigned char)(100 + 155 * nx),  // R
                            (unsigned char)(150 + 105 * nx),  // G
                            (unsigned char)(255),             // B
                            255                              // A
                        };
                        break;
                    case 1: // Left face
                        color = {
                            (unsigned char)(100 + 155 * (1.0f - nx)),  // R
                            (unsigned char)(150 + 105 * (1.0f - nx)),  // G
                            (unsigned char)(255),                      // B
                            255                                       // A
                        };
                        break;
                    case 2: // Top face
                        color = {
                            (unsigned char)(100 + 155 * ny),  // R
                            (unsigned char)(150 + 105 * ny),  // G
                            (unsigned char)(255),             // B
                            255                              // A
                        };
                        break;
                    case 3: // Bottom face
                        color = {
                            (unsigned char)(100 + 155 * (1.0f - ny)),  // R
                            (unsigned char)(150 + 105 * (1.0f - ny)),  // G
                            (unsigned char)(255),                      // B
                            255                                       // A
                        };
                        break;
                    case 4: // Front face
                        color = {
                            (unsigned char)(100 + 155 * nx),  // R
                            (unsigned char)(150 + 105 * nx),  // G
                            (unsigned char)(255),             // B
                            255                              // A
                        };
                        break;
                    case 5: // Back face
                        color = {
                            (unsigned char)(100 + 155 * (1.0f - nx)),  // R
                            (unsigned char)(150 + 105 * (1.0f - nx)),  // G
                            (unsigned char)(255),                      // B
                            255                                       // A
                        };
                        break;
                }
                
                ImageDrawPixel(&faces[face], x, y, color);
            }
        }
    }
    
    // Create a single image with all faces arranged in a cross pattern
    Image skybox = GenImageColor(1536, 1024, BLANK);  // 3x2 arrangement of 512x512 faces
    
    // Copy each face to the appropriate position in the skybox image
    ImageDraw(&skybox, faces[0], Rectangle{0, 512, 512, 512}, Rectangle{0, 0, 512, 512}, WHITE);  // Right
    ImageDraw(&skybox, faces[1], Rectangle{0, 512, 512, 512}, Rectangle{512, 0, 512, 512}, WHITE);  // Left
    ImageDraw(&skybox, faces[2], Rectangle{0, 512, 512, 512}, Rectangle{1024, 0, 512, 512}, WHITE);  // Top
    ImageDraw(&skybox, faces[3], Rectangle{0, 512, 512, 512}, Rectangle{1024, 512, 512, 512}, WHITE);  // Bottom
    ImageDraw(&skybox, faces[4], Rectangle{0, 512, 512, 512}, Rectangle{512, 512, 512, 512}, WHITE);  // Front
    ImageDraw(&skybox, faces[5], Rectangle{0, 512, 512, 512}, Rectangle{0, 512, 512, 512}, WHITE);  // Back
    
    // Save the image
    ExportImage(skybox, "resources/textures/skybox.png");
    
    // Cleanup
    for (int i = 0; i < 6; i++) {
        UnloadImage(faces[i]);
    }
    UnloadImage(skybox);
    CloseWindow();
    
    printf("Skybox texture created successfully!\n");
    return 0;
} 