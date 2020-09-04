# 3D Slicer's extension build tool defines CollageRadiomics_BUILD_SLICER_EXTENSION:BOOL=ON
# to indicate that this project is being built as a 3D Slicer extension.
if (CollageRadiomics_BUILD_SLICER_EXTENSION)
  message(STATUS "Building project as a Slicer extension")
  project(CollageRadiomics)
  add_subdirectory(slicer)
  return()
endif()
