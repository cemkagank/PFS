# Define the compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -Wall -std=c++11 -fopenmp -Iinclude -lGL -lraylib -O2

# Define the target executable (inside bin/)
TARGET = bin/fluids

# Define the source files (with paths)
SRCS = src/main.cpp src/Window.cpp  src/Engine.cpp src/Particle.cpp src/imgui_draw.cpp src/imgui_widgets.cpp src/imgui_tables.cpp src/imgui.cpp src/rlImGui.cpp

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
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile each .cpp file into a .o file in the obj/ directory
$(OBJDIR)/%.o: src/%.cpp
	@mkdir -p $(OBJDIR)  # Create the obj/ directory if it doesn't exist
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the project (remove object files and executable)
clean:
	rm -rf $(OBJDIR) $(BINDIR)

run:
	./$(TARGET)

make:
	make clean
	make
	make run
# Phony targets (not associated with files)
.PHONY: all clean
