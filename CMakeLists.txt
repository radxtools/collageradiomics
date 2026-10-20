cmake_minimum_required(VERSION 3.13.4)
project(CollageRadiomics)

# Extension metadata is kept at the repository root for catalog validation.
set(EXTENSION_HOMEPAGE "https://github.com/radxtools/collageradiomics/tree/master/slicer/collageradiomics")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Pallavi Tiwari (BriC), Satish Viswanath (BriC), Rob Toth (Toth Technology), Nathan Hillyer (Toth Technology)")
set(EXTENSION_DESCRIPTION "Voxel-wise CoLlAGe radiomics from local gradient orientations, with labelmap and segmentation inputs and thirteen texture features.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/radxtools/collageradiomics/master/slicer/collageradiomics/CollageRadiomicsSlicer/Resources/Icons/CollageRadiomicsSlicer.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/radxtools/collageradiomics/master/slicer/collageradiomics/CollageRadiomicsSlicer/Screenshot.jpg")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/slicer/collageradiomics/CollageRadiomicsSlicer" CollageRadiomicsSlicer)
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
