#!/bin/bash
# Compile all GLSL compute shaders to SPIR-V
# Requires: glslangValidator (from Vulkan SDK or `pacman -S glslang`)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

for comp in *.comp; do
    spv="${comp%.comp}.spv"
    echo "Compiling $comp -> $spv"
    glslangValidator -V "$comp" -o "$spv"
done

echo "Done. $(ls *.spv | wc -l) shaders compiled."
