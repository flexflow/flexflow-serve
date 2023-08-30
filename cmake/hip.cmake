if (NOT FF_HIP_ARCH STREQUAL "")
    if (FF_HIP_ARCH STREQUAL "all")
        #gfx700,gfx701,gfx702,gfx703,gfx704,gfx705,gfx801,gfx802,gfx803,gfx805,
        set(FF_HIP_ARCH "gfx900,gfx902,gfx904,gfx906,gfx908,gfx909,gfx90a,gfx90c,gfx940,gfx1010,gfx1011,gfx1012,gfx1013,gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103")
    endif()
    string(REPLACE "," " " HIP_ARCH_LIST "${FF_HIP_ARCH}")
endif()

message(STATUS "FF_HIP_ARCH: ${FF_HIP_ARCH}")
if(FF_GPU_BACKEND STREQUAL "hip_rocm")
    set(HIP_CLANG_PATH ${ROCM_PATH}/llvm/bin CACHE STRING "Path to the clang compiler by ROCM" FORCE)
endif()
