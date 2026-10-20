cmake_minimum_required(VERSION 3.13.4)
project(CollageRadiomics)

# Slicer's extension builders configure the repository root. Keep the extension
# sources together while also supporting direct builds from its subdirectory.
include("${CMAKE_CURRENT_SOURCE_DIR}/slicer/collageradiomics/CMakeLists.txt")
