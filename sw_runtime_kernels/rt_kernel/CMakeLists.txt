#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.20)

project(rt_kernel)


option(ENABLE_MANAGEMENT_KERNEL_BUILD "Enable management kernel build" OFF)

set(prebuild_binary_dir "${CMAKE_CURRENT_SOURCE_DIR}/prebuild")
set(target_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/rt_kernel_bin")
file(MAKE_DIRECTORY ${target_binary_dir})

set(elf_suffix ".elf")
set(obj_suffix ".o")
set(kernels_to_build)

add_custom_target(rt_kernel_build)

if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    if(NOT DEFINED MV_TOOLS_PATH AND DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
        set(MV_TOOLS_PATH $ENV{MV_TOOLS_DIR}/$ENV{MV_TOOLS_VERSION})
    endif()

    exists_mv_tools_version(available)
    if(NOT available)
        get_mv_tools()
    endif()

    get_mv_tools_path(MV_TOOLS_PATH)

    if(ENABLE_NPU_MONO)
        if(DEFINED ENV{FIRMWARE_VPU_DIR})
            message(AUTHOR_WARNING "FIRMWARE_VPU_DIR environment variable is deprecated when ENABLE_NPU_MONO=ON")
        endif()
        if(NOT DEFINED NPU_MONO_FIRMWARE_PROJECT_DIR)
            message(FATAL_ERROR "Firmware project dir is not set while `npu_mono` was activated")
        endif()
        set(FIRMWARE_VPU_DIR "${NPU_MONO_FIRMWARE_PROJECT_DIR}")
    else()
        if(NOT DEFINED ENV{FIRMWARE_VPU_DIR})
            message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined when ENABLE_MANAGEMENT_KERNEL_BUILD=ON")
        endif()
        set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")
    endif()

    if(UNIX)
        set(mv_tools_compile "${MV_TOOLS_PATH}/linux64/bin/moviCompile")
        set(mv_tools_link "${MV_TOOLS_PATH}/linux64/bin/moviLLD")
    elseif(WIN32)
        set(mv_tools_compile "${MV_TOOLS_PATH}/win64/bin/moviCompile.exe")
        set(mv_tools_link "${MV_TOOLS_PATH}/win64/bin/moviLLD.exe")
    else()
        message(FATAL_ERROR "Unsupported operating system")
    endif()

    set(link_libraries_list_VPU3720
        "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibm.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibc_lite.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibcrt.a"
    )
    set(link_libraries_list_VPU4000
        "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibm.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibc_lite.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibcrt.a"
    )
    set(link_libraries_list_VPU5000
        "${MV_TOOLS_PATH}/common/moviCompile/lib/50xxxx/mlibm.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/50xxxx/mlibc_lite.a"
        "${MV_TOOLS_PATH}/common/moviCompile/lib/50xxxx/mlibcrt.a"
    )
    # Platform list: each entry maps a kernel CPU number to the firmware source subdirectory.
    # 8000 shares the 7000 firmware sources.
    set(_rt_subdir_3720 3720)
    set(_rt_subdir_4000 4000)
    set(_rt_subdir_5000 5000)
    set(_rt_platforms 3720 4000 5000)
    # Shorthand for the slimfw act_runtime base path used by 5000+
    set(_slimfw_ar "${FIRMWARE_VPU_DIR}/slimfw/validation/validationApps/system/nn/validation/act_runtime")

foreach(platform IN LISTS _rt_platforms)
    set(platform_subdir   ${_rt_subdir_${platform}})
    set(kernel_cpu        "${platform}xx")
    set(kernel_cpu_suffix ".${platform}xx")
    set(kernel_src        "nnActEntry.cpp")
    set(elf_file          "nnActEntry${kernel_cpu_suffix}${elf_suffix}")
    set(elf_path          "${target_binary_dir}/${elf_file}")
    set(link_script_file  "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_rt_kernel.ld")

    if(platform STREQUAL "3720")
        # The 3720 management kernel is not rebuilt from sources; use the prebuilt binary.
        set(kernel_entry "nnActEntry")
        set(optimization_opts "-O3")
        set(kernel_src_dir "${FIRMWARE_VPU_DIR}/vpu2/system/nn/act_runtime/src/3720")
        set(link_libraries_list ${link_libraries_list_VPU3720})
        set(asm_include_list)
        set(define_symbols_list "__act_shave__")
        set(include_dirs_list
            "${CMAKE_CURRENT_SOURCE_DIR}/inc"
            "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/include"
            "${FIRMWARE_VPU_DIR}/vpu2/include/nn"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/errors/errorCodes/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/hardware/registerMap/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/hardware/utils/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/nn/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/resource/barrier/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/shave/svuCtrl_3600/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/shave/svuL1c/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/shave/svuShared_3600/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/vcpr/perf_timer"
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/memory/dma/inc/"
            "${FIRMWARE_VPU_DIR}/vpu2/system/supervision/errorManager/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/system/shave/svuCtrl_3600/inc"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/act_runtime/inc/3720"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common_runtime/inc/3720"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common_runtime/inc/common"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common/inc/3720"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common/inc/common"
            "${FIRMWARE_VPU_DIR}/vpu2/utils/common/inc"
        )
        set(extra_src_list
            "${FIRMWARE_VPU_DIR}/vpu2/drivers/nn/src/nn_fifo.cpp"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common_runtime/src/3720/nn_fifo_manager.cpp"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common_runtime/src/common/nn_fifo_manager.cpp"
            "${FIRMWARE_VPU_DIR}/vpu2/system/nn/common_runtime/src/common/nn_perf_manager.cpp"
        )
        set(asm_src_list
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/3720/preemption.asm"
        )
        continue()

    elseif(platform STREQUAL "4000")
        set(kernel_entry "nn_shave_entry")
        set(optimization_opts "-O2" "-fforce-enable-int128" "-fno-jump-tables")
        set(kernel_src_dir "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/4000")
        set(link_libraries_list ${link_libraries_list_VPU4000})
        set(asm_include_list "-Xassembler" "-i" "-Xassembler"
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/common")
        set(define_symbols_list "__act_shave__" "CONFIG_TARGET_SOC_4000")
        set(include_dirs_list
            "${CMAKE_CURRENT_SOURCE_DIR}/inc"
            "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc"
            "${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc"
            "${FIRMWARE_VPU_DIR}/include"
            "${FIRMWARE_VPU_DIR}/include/shave"
            "${FIRMWARE_VPU_DIR}/include/vpu_c_header"
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/inc/common"
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/inc/4000"
            "${FIRMWARE_VPU_DIR}/include/nn"
            "${FIRMWARE_VPU_DIR}/drivers/barriers"
            "${FIRMWARE_VPU_DIR}/system/nn/common/inc/4000"
            "${FIRMWARE_VPU_DIR}/system/nn/common/inc/common"
            "${FIRMWARE_VPU_DIR}/system/nn/common_runtime/inc/4000"
            "${FIRMWARE_VPU_DIR}/system/nn/common_runtime/inc/common"
            "${FIRMWARE_VPU_DIR}/system/nn/include"
            "${FIRMWARE_VPU_DIR}/drivers/nn/inc/common"
            "${FIRMWARE_VPU_DIR}/drivers/nn/inc/4000"
        )
        set(extra_src_list
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/4000/nnActRtFuncs.cpp"
            "${FIRMWARE_VPU_DIR}/system/nn/common_runtime/src/common/nn_perf_manager.cpp"
            "${FIRMWARE_VPU_DIR}/drivers/nn/src/4000/nn_fifo.cpp"
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/common/nnActCache.cpp"
            "${FIRMWARE_VPU_DIR}/drivers/shave/cache/ShaveL1Cache-Shave.c"
        )
        set(asm_src_list
            "${FIRMWARE_VPU_DIR}/system/nn/act_runtime/src/4000/preemption.S"
        )

    else()
        # 5000+ slimfw family: share common build settings and source structure.
        set(kernel_entry "nn_shave_entry")
        set(optimization_opts "-O2" "-fforce-enable-int128" "-fno-jump-tables")
        set(kernel_src_dir "${_slimfw_ar}/src/common")
        set(asm_include_list "-Xassembler" "-i" "-Xassembler" "${_slimfw_ar}/src/common")

        if(platform STREQUAL "5000")
            set(define_symbols_list "__act_shave__" "CONFIG_TARGET_SOC_4000" "CONFIG_TARGET_SOC_5000")
            set(link_libraries_list ${link_libraries_list_VPU5000})
        endif()

        set(include_dirs_list
            "${CMAKE_CURRENT_SOURCE_DIR}/inc"
            "${FIRMWARE_VPU_DIR}/slimfw/drivers/hardware/registerMap/inc"
            "${FIRMWARE_VPU_DIR}/slimfw/include"
            "${FIRMWARE_VPU_DIR}/slimfw/include/shave"
            "${FIRMWARE_VPU_DIR}/slimfw/include/vpu_c_header"
            "${_slimfw_ar}/inc/common"
            "${_slimfw_ar}/inc/${platform_subdir}"
            "${FIRMWARE_VPU_DIR}/slimfw/include/nn"
            "${FIRMWARE_VPU_DIR}/slimfw/drivers/barriers"
            "${FIRMWARE_VPU_DIR}/slimfw/system/nn/common/inc"
            "${FIRMWARE_VPU_DIR}/slimfw/system/nn/include"
            "${FIRMWARE_VPU_DIR}/slimfw/drivers/nn/inc"
        )

        set(extra_src_list
            "${_slimfw_ar}/src/${platform_subdir}/nnActRtFuncs.cpp"
            "${_slimfw_ar}/src/common/nnActPerfManager.cpp"
            "${_slimfw_ar}/src/common/nnActCache.cpp"
            "${FIRMWARE_VPU_DIR}/slimfw/drivers/nn/src/nn_fifo.cpp"
            "${FIRMWARE_VPU_DIR}/slimfw/drivers/shave/cache/ShaveL1Cache-Shave.c"
        )
        set(asm_src_list
            "${_slimfw_ar}/src/${platform_subdir}/preemption.S"
            "${_slimfw_ar}/src/${platform_subdir}/entry.S"
        )
    endif()

    ### RESOLVE FILE NAMES ###
    set(kernel_src_file "${kernel_src_dir}/${kernel_src}")
    if(IS_ABSOLUTE "${kernel_src_file}")
        set(kernel_src_path "${kernel_src_file}")
    else()
        set(kernel_src_path "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_src_file}")
    endif()

    set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")
    set(dep_file "${kernel_src}${kernel_cpu_suffix}.d")

    list(TRANSFORM include_dirs_list PREPEND "-I")
    list(TRANSFORM define_symbols_list PREPEND "-D")

    set(compile_options
        "-mcpu=${kernel_cpu}"
        ${optimization_opts}
        "-fno-exceptions"
        "-ffunction-sections"
        "-fdata-sections"
        ${include_dirs_list}
        ${define_symbols_list}
        "-mshave-preemption-checks=off")

    add_custom_command(
        OUTPUT "${obj_file}"
        DEPENDS "${kernel_src_path}"
        DEPFILE "${dep_file}"
        COMMAND "${mv_tools_compile}" -MD ${compile_options} -c "${kernel_src_path}" -o "${obj_file}"
    )

    set(obj_file_list "${obj_file}")

    if(extra_src_list)
        foreach(extra_src_file ${extra_src_list})
            get_filename_component(src_name ${extra_src_file} NAME_WE)
            get_filename_component(dir_name ${extra_src_file} DIRECTORY)
            get_filename_component(dir_name ${dir_name} NAME_WE)

            set(obj_path "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}")
            file(MAKE_DIRECTORY "${obj_path}")
            set(base_name "${obj_path}/${src_name}.${kernel_cpu}")
            set(obj_file "${base_name}${obj_suffix}")
            set(dep_file "${base_name}.d")

            add_custom_command(
                OUTPUT "${obj_file}"
                DEPENDS "${extra_src_file}"
                DEPFILE "${dep_file}"
                COMMAND "${mv_tools_compile}" -MD ${compile_options} -c "${extra_src_file}" -o "${obj_file}"
            )
            list(APPEND obj_file_list "${obj_file}")
        endforeach()
    endif()

    if(asm_src_list)
        set(inc_list)
        set(_asm_inc_dirs "${kernel_src_dir}")
        foreach(_asm_src IN LISTS asm_src_list)
            get_filename_component(_asm_dir "${_asm_src}" DIRECTORY)
            list(APPEND _asm_inc_dirs "${_asm_dir}")
        endforeach()
        list(REMOVE_DUPLICATES _asm_inc_dirs)
        foreach(inc_dir IN LISTS _asm_inc_dirs)
            file(GLOB_RECURSE inc_files CONFIGURE_DEPENDS "${inc_dir}/*.inc" "${inc_dir}/*.S")
            list(APPEND inc_list ${inc_files})
        endforeach()

        foreach(asm_src_file ${asm_src_list})
            get_filename_component(src_name ${asm_src_file} NAME_WE)
            get_filename_component(dir_name ${asm_src_file} DIRECTORY)
            get_filename_component(dir_name ${dir_name} NAME_WE)

            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}")
            set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")

            add_custom_command(
                OUTPUT "${obj_file}"
                DEPENDS "${asm_src_file}" "${inc_list}"
                COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${asm_include_list} -c "${asm_src_file}" -o "${obj_file}"
            )
            list(APPEND obj_file_list "${obj_file}")
        endforeach()
    endif()

    set(link_options
        "--script" "${link_script_file}"
        "-entry" "${kernel_entry}"
        "--gc-sections"
        "--strip-debug"
        "--discard-all"
        "-zmax-page-size=16")

    add_custom_command(
        OUTPUT "${elf_path}"
        DEPENDS ${obj_file_list} "${link_script_file}" ${link_libraries_list}
        COMMAND "${mv_tools_link}" ${link_options} ${obj_file_list} -EL ${link_libraries_list} --output "${elf_path}"
    )
    list(APPEND kernels_to_build "${elf_path}")
    add_custom_target("${elf_file}" DEPENDS "${elf_path}")
    add_dependencies(rt_kernel_build "${elf_file}")

    endforeach()
endif()

### UPDATE PREBUILT KERNELS ###
file(GLOB prebuild_kernels RELATIVE "${prebuild_binary_dir}" CONFIGURE_DEPENDS "${prebuild_binary_dir}/*.elf")

if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    if(NOT kernels_to_build)
        message(FATAL_ERROR "No management kernels to build")
    endif()

    set(update_prebuilt_commands)
    foreach(kernel_file IN LISTS kernels_to_build)
        list(APPEND update_prebuilt_commands
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${kernel_file}" "${prebuild_binary_dir}")
    endforeach()
    add_custom_target(update_rt_kernel_prebuilt_binaries
        DEPENDS rt_kernel_build
        ${update_prebuilt_commands}
        COMMENT "Updating prebuilt management kernels"
    )
else()
    add_custom_target(update_rt_kernel_prebuilt_binaries)  # dummy target if not building
endif()

### LIBRARY TARGET ###
list(TRANSFORM prebuild_kernels PREPEND "${prebuild_binary_dir}/" OUTPUT_VARIABLE prebuild_bins)

if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    set(_rt_kernel_gen_deps rt_kernel_build update_rt_kernel_prebuilt_binaries ${kernels_to_build})
else()
    set(_rt_kernel_gen_deps)
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_rt_kernel_binary_resources.cpp"
    COMMAND ${CMAKE_COMMAND} -D KERNELS_BIN_DIR="${prebuild_binary_dir}"
                             -D GENERATED_FILE="generated_rt_kernel_binary_resources.cpp"
                             -D MAP_NAME="rtKernelBinaryResourcesMap"
                             -D EXTENSION="${elf_suffix}"
                             -P "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/cmake/embed_shave_binaries.cmake"
    DEPENDS ${_rt_kernel_gen_deps} ${prebuild_bins}
)

add_library(rt_kernel_lib OBJECT
    "${CMAKE_CURRENT_BINARY_DIR}/generated_rt_kernel_binary_resources.cpp"
)
target_compile_options(rt_kernel_lib PRIVATE -O0)
