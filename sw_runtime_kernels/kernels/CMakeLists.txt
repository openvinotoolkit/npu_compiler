# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

cmake_minimum_required(VERSION 3.20)

project(kernels)

option(ENABLE_SHAVE_BINARIES_BUILD "Enable shave binaries build, if disabled, prebuilt binaries will be used" OFF)
option(ENABLE_MANAGEMENT_KERNEL_BUILD "Enable management kernel build" OFF)
option(ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD "Enable firmware.vpu.client sources kernels build" OFF)

set(target_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin")
file(MAKE_DIRECTORY ${target_binary_dir})

# Establish if we want to build the kernels or just use prebuilts
if(NOT (ENABLE_SHAVE_BINARIES_BUILD OR ENABLE_MANAGEMENT_KERNEL_BUILD OR ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD))
  set(build_kernels FALSE)
else()
  set(build_kernels TRUE)
endif()

# Defines section and general use code
set(obj_suffix ".o")
set(elf_suffix ".elf")
set(kernel_descrip_dir "${CMAKE_CURRENT_SOURCE_DIR}/descrip/")
set(kernel_descrip_list "")
set(act_shave_kernels "")

add_custom_target(act_shave_kernels_ready)
add_custom_target(act_shave_kernels_build)

# Enable building kernels from separate build folder if needed
if(ENABLE_MLIR_COMPILER AND build_kernels)
  exists_mv_tools_version(available)
  if(NOT available)
    get_mv_tools()
  endif()

  get_mv_tools_path(MV_TOOLS_PATH)
else()
  set(MV_TOOLS_PATH "$ENV{IE_NPU_MV_TOOLS_PATH}")
endif()

# Delete previously build binaries from the build folder to avoid copy of old binaries
file(REMOVE_RECURSE "${target_binary_dir}/*${elf_suffix}")
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/*${obj_suffix}")
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp")

if(build_kernels)
  if(UNIX)
    set(mv_tools_compile "${MV_TOOLS_PATH}/linux64/bin/moviCompile")
    set(mv_tools_link "${MV_TOOLS_PATH}/linux64/bin/moviLLD")
  elseif(WIN32)
    set(mv_tools_compile "${MV_TOOLS_PATH}/win32/bin/moviCompile.exe")
    set(mv_tools_link "${MV_TOOLS_PATH}/win64/bin/moviLLD.exe")
  else()
    message(FATAL_ERROR "Unsupported operating system")
  endif()

  set(link_libraries_list_VPU3720
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibm.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibc_lite.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibc_lite_lgpl.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibcrt.a"
  )

  set(link_libraries_list_VPU4000
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibm.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibc_lite.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibc_lite_lgpl.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibcrt.a"
  )
endif()

# Main code block
# Only in case of binaries build we run through the whole descrip list
if(ENABLE_SHAVE_BINARIES_BUILD)
  file(GLOB kernel_list "${kernel_descrip_dir}*.txt")
  list(APPEND kernel_descrip_list ${kernel_list})
endif()

if(ENABLE_MANAGEMENT_KERNEL_BUILD OR ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD)
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
        message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined when ENABLE_MANAGEMENT_KERNEL_BUILD=ON or ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD=ON")
    endif()

    set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")
  endif()

  # If we want to build only MGMT kernels we should iterate only through those descrip files
  if(ENABLE_MANAGEMENT_KERNEL_BUILD AND NOT ENABLE_SHAVE_BINARIES_BUILD)
    file(GLOB mgmt_kernel_list "${kernel_descrip_dir}nnActEntry_*.txt")
    list(APPEND kernel_descrip_list ${mgmt_kernel_list})
  endif()

  # If we want to build only kernels with firmware sources we should iterate only through those descrip files
  if(ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD AND NOT ENABLE_SHAVE_BINARIES_BUILD)
    file(GLOB firmware_kernel_list "${kernel_descrip_dir}*_fw.txt")
    list(APPEND kernel_descrip_list ${firmware_kernel_list})
  endif()

endif()

# Main iterator
foreach(kernel_descrip ${kernel_descrip_list})
  # Initial list of parameters, will be populated after we read descrip files
  set(kernel_entry "")
  set(kernel_src_dir "src")
  set(optimization_opts "-O3")
  set(include_dirs_list "")
  set(define_symbols_list "")
  set(cppflags_list "")
  set(always_inline "no")
  set(rt_kernel "no")
  set(firmware_kernel "no")
  set(extra_src_list "")
  set(asm_src_list "")
  set(asm_include_list "")
  set(link_script_file "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_kernel.ld")

  # Reading descrip files one by one from the list created above
  get_filename_component(kernel_descrip_path ${kernel_descrip} DIRECTORY)

  include("${kernel_descrip}")
  get_filename_component(kernel_name ${kernel_src} NAME_WE)

  if(kernel_entry STREQUAL "")
    set(kernel_entry "${kernel_name}")
  endif()

  if(build_kernels)
    if(kernel_cpunum STREQUAL "3720")
      list(APPEND define_symbols_list "USE_3720_INSTRUCTIONS")
      set(link_libraries_list ${link_libraries_list_VPU3720})
      if(rt_kernel STREQUAL "yes")
        list(APPEND define_symbols_list "CONFIG_TARGET_SOC_3720")
      endif()
    elseif(kernel_cpunum STREQUAL "4000")
      list(APPEND define_symbols_list "USE_4000_INSTRUCTIONS")
      set(link_libraries_list ${link_libraries_list_VPU4000})
      if(rt_kernel STREQUAL "yes")
        list(APPEND define_symbols_list "CONFIG_TARGET_SOC_4000")
      endif()
    endif()

    if(always_inline STREQUAL "yes")
      list(APPEND define_symbols_list "CONFIG_ALWAYS_INLINE")
    endif()

    list(INSERT include_dirs_list 0 "${CMAKE_CURRENT_SOURCE_DIR}/inc")
    list(INSERT define_symbols_list 0 "__act_shave__")
  endif()

  set(kernel_cpu "${kernel_cpunum}xx")
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")
  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")

  set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")
  set(elf_file "${kernel_name}${kernel_cpu_suffix}${elf_suffix}")
  set(elf_path "${target_binary_dir}/${elf_file}")
  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
    set(obj_file_lsu0_wo "${kernel_src}${kernel_cpu_suffix}_lsu0_wo${obj_suffix}")
    set(elf_file_lsu0_wo "${kernel_name}${kernel_cpu_suffix}_lsu0_wo${elf_suffix}")
    set(cppflags_list_lsu0_wo
      "-mllvm"
      "-shave-lsu-load-policy=use-only-lsu1"
      "-mllvm"
      "-shave-lsu-store-policy=prefer-lsu0"
    )
    set(elf_path_lsu0_wo "${target_binary_dir}/${elf_file_lsu0_wo}")
  endif()

  if(build_kernels)
    if(rt_kernel STREQUAL "yes")
     if(NOT ENABLE_MANAGEMENT_KERNEL_BUILD)
      add_custom_target("${kernel_name}${kernel_cpu_suffix}_kernel_copy"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${elf_file}" "${target_binary_dir}"
      )

      add_dependencies(act_shave_kernels_build "${kernel_name}${kernel_cpu_suffix}_kernel_copy")
      continue()
    endif()

    set(kernel_src_path "${kernel_src_file}")
    # ACT management kernel must be built with Shave preemption checks disabled
    set(shave_preemption_opt "-mshave-preemption-checks=off")
    set(link_script_file "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_rt_kernel.ld")
  else()
    set(kernel_src_path "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_src_file}")
      set(shave_preemption_opt "-mshave-preemption-checks=restore")
      list(APPEND shave_preemption_opt "-mshave-low-impact-preemption" )
      list(APPEND shave_preemption_opt "-mshave-preemption-max-loop-depth=1")
    endif()
  endif()

  # List needed to create the final embedded description file
  list(APPEND act_shave_kernels ${kernel_src_file})

  if(build_kernels)
    # List needed to create the final embedded description file
    list(TRANSFORM include_dirs_list PREPEND "-I")
    list(TRANSFORM define_symbols_list PREPEND "-D")

    # Compile the kernel and output an object file
    add_custom_command(
      OUTPUT "${obj_file}"
      DEPENDS "${kernel_src_file}"
      COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${cppflags_list} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${kernel_src_path}" -o "${obj_file}"
    )

    set(obj_file_list "${obj_file}")

    if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
      add_custom_command(
        OUTPUT "${obj_file_lsu0_wo}"
        DEPENDS "${kernel_src_file}"
        COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${cppflags_list} ${cppflags_list_lsu0_wo} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${kernel_src_path}" -o "${obj_file_lsu0_wo}"
      )

      set(obj_file_list_lsu0_wo "${obj_file_lsu0_wo}")
    endif()

    # Compile extra sources if specified in the descrip file
    if(NOT extra_src_list STREQUAL "")
      foreach(extra_src_file ${extra_src_list})
        get_filename_component(src_name ${extra_src_file} NAME_WE)
        get_filename_component(dir_name ${extra_src_file} DIRECTORY)

        # Discard full path, just keep name of the dir the file is in
        get_filename_component(dir_name ${dir_name} NAME_WE)

        # Some extra_src files have the same filename so the .o files must be in separate directories to avoid overwriting.
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name})
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")

        add_custom_command(
          OUTPUT "${obj_file}"
          DEPENDS "${kernel_src_file}"
          COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${cppflags_list} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${extra_src_file}" -o "${obj_file}"
        )

        list(APPEND obj_file_list "${obj_file}")
        if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
          list(APPEND obj_file_list_lsu0_wo "${obj_file}")
        endif()
      endforeach()
    endif()

    # Compile asm files if specified in the descrip file
    if(NOT asm_src_list STREQUAL "")
      foreach(asm_src_file ${asm_src_list})
        get_filename_component(src_name ${asm_src_file} NAME_WE)
        get_filename_component(dir_name ${asm_src_file} DIRECTORY)

        # Discard full path, just keep name of the dir the file is in
        get_filename_component(dir_name ${dir_name} NAME_WE)

        # Some asm_src files have the same filename so the .o files must be in separate directories to avoid overwriting.
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name})
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")

        add_custom_command(
          OUTPUT "${obj_file}"
          DEPENDS "${kernel_src_file}"
          COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${asm_include_list} -c "${asm_src_file}" -o "${obj_file}"
        )

        list(APPEND obj_file_list "${obj_file}")
        if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
          list(APPEND obj_file_list_lsu0_wo "${obj_file}")
        endif()
      endforeach()
    endif()

    # Link the sources, add entry point and windowed sections, then dump the elf file
    add_custom_command(
      OUTPUT "${elf_path}"
      DEPENDS ${obj_file_list}
      COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list} --output "${elf_path}"
    )

    if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
      add_custom_command(
        OUTPUT "${elf_path_lsu0_wo}"
        DEPENDS ${obj_file_list_lsu0_wo}
        COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list_lsu0_wo} -EL ${link_libraries_list} --output "${elf_path_lsu0_wo}"
      )
    endif()

    # Delete extra and asm sources after compile such that we can apply different optimizations if needed
    if(NOT extra_src_list STREQUAL "")
      foreach(extra_src_file ${extra_src_list})
        get_filename_component(src_name ${extra_src_file} NAME_WE)
        file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")
      endforeach()
    endif()

    if(NOT asm_src_list STREQUAL "")
      foreach(asm_src_file ${asm_src_list})
        get_filename_component(src_name ${asm_src_file} NAME_WE)
        file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")
      endforeach()
    endif()
  endif()

  add_custom_target("${elf_file}" DEPENDS "${elf_path}")
  add_dependencies(act_shave_kernels_build "${elf_file}")

  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel STREQUAL "yes") # MTL DDR split access workaround
    add_custom_target("${elf_file_lsu0_wo}" DEPENDS "${elf_path_lsu0_wo}")
    add_dependencies(act_shave_kernels_build "${elf_file_lsu0_wo}")
  endif()
endforeach()

# If we don't want to rebuild anything just copy the prebuilts and create the embedded sections file
if(build_kernels)
  # Copy binaries after compile
  add_custom_target(act_shave_kernels_copy
    DEPENDS ${act_shave_kernels}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${target_binary_dir}/" "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/"
  )
  add_dependencies(act_shave_kernels_copy act_shave_kernels_build)
  add_dependencies(act_shave_kernels_ready act_shave_kernels_copy)
else()
  add_custom_target(update_prebuilt_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin" "${target_binary_dir}"
  )
  add_dependencies(act_shave_kernels_ready act_shave_kernels_build update_prebuilt_binaries)
endif()

# Add file-level and target-level dependencies
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/embed_shave_binaries.cmake"
  DEPENDS ${act_shave_kernels} act_shave_kernels_ready
)

add_library(act_shave_kernels_lib OBJECT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp")
# The library contains many large arrays, because of which it compiles slowly if the compiler optimisations are enabled.
target_compile_options(act_shave_kernels_lib PRIVATE -O0)
