# Define the compiler
CXX = g++

# Define compiler flags and include paths
CXXFLAGS = -Wall -g -std=gnu++17 -fopenmp -Iinclude -I/usr/local/lib -O2

# Define linker flags and link the raylib library from /usr/local/lib
LDFLAGS = -L/usr/local/lib -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

# Define the target executable (inside bin/)
TARGET = bin/fluids

# Define the source files (with paths)
SRCS = src/main.cpp src/Window.cpp src/Engine.cpp  src/imgui_draw.cpp src/imgui_widgets.cpp src/imgui_tables.cpp src/imgui.cpp src/rlImGui.cpp

# Define the object directory
OBJDIR = obj

# Define the binary directory
BINDIR = bin

# Define the object files (placing them in the obj/ directory)
OBJS = $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(SRCS))

# Default rule to build the target
all: $(TARGET)

# Rule to link the object files and create the executable in the bin/ directory
$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)  # Create the bin/ directory if it doesn't exist
	$(CXX) -o $(TARGET) $(OBJS) $(CXXFLAGS) $(LDFLAGS)

# Rule to compile each .cpp file into a .o file in the obj/ directory, and generate dependency files
$(OBJDIR)/%.o: src/%.cpp
	@mkdir -p $(OBJDIR)  # Create the obj/ directory if it doesn't exist
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Include the generated dependency files
-include $(OBJS:.o=.d)

# Clean up the project (remove object files and executable)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Run the executable after building
run:
	make all
	./$(TARGET)

# Phony targets (not associated with files)
.PHONY: all clean run
