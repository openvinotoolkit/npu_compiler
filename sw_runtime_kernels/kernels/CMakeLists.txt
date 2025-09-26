# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

cmake_minimum_required(VERSION 3.20)

project(kernels)


option(ENABLE_SHAVE_BINARIES_BUILD "Enable shave binaries build, if disabled, prebuilt binaries will be used" OFF)
option(ENABLE_MANAGEMENT_KERNEL_BUILD "Enable management kernel build" OFF)
option(ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD "Enable firmware.vpu.client sources kernels build" OFF)
option(ENABLE_SHAVE_BITCODE_BUILD "Embed kernel bitcode into the build (experimental)" OFF)

set(prebuild_binary_dir "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin")
set(target_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin")
file(MAKE_DIRECTORY ${target_binary_dir})
set(target_asm_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_asm")
file(MAKE_DIRECTORY ${target_asm_dir})
set(target_bitcode_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bitcode")
file(MAKE_DIRECTORY ${target_bitcode_dir})
set(target_asm_archive_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_asm_archive")
file(MAKE_DIRECTORY ${target_asm_archive_dir})

# Establish if we want to build the kernels or just use prebuilts
if(ENABLE_SHAVE_BINARIES_BUILD OR ENABLE_MANAGEMENT_KERNEL_BUILD OR ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD)
  set(build_kernels TRUE)
else()
  set(build_kernels FALSE)
endif()

if (build_kernels AND ENABLE_SHAVE_BITCODE_BUILD)
  set(build_bitcode TRUE)
else()
  set(build_bitcode FALSE)
endif()

# Defines section and general use code
set(asm_suffix ".s")
set(obj_suffix ".o")
set(elf_suffix ".elf")
set(bitcode_suffix ".bc")
set(linked_bitcode_suffix ".lbc")
set(ar_suffix ".a")
set(kernel_descrip_dir "${CMAKE_CURRENT_SOURCE_DIR}/descrip")
set(kernel_descrip_list)
set(all_kernels)
set(kernels_to_build)

add_custom_target(act_shave_kernels_build)
add_custom_target(act_shave_kernels_asm)

if(NOT DEFINED MV_TOOLS_PATH AND DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
    set(MV_TOOLS_PATH $ENV{MV_TOOLS_DIR}/$ENV{MV_TOOLS_VERSION})
endif()

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

if(build_kernels)
  if(UNIX)
    set(mv_tools_compile "${MV_TOOLS_PATH}/linux64/bin/moviCompile")
    set(mv_tools_link "${MV_TOOLS_PATH}/linux64/bin/moviLLD")
    if(build_bitcode)
      # Assume these are in the environment until they can be shipped
      # with as part of the movicompile package.
      set(mv_tools_llvm_link "llvm-link")
      set(mv_tools_llvm_ar "llvm-ar")
    endif()
  elseif(WIN32)
    set(mv_tools_compile "${MV_TOOLS_PATH}/win32/bin/moviCompile.exe")
    set(mv_tools_link "${MV_TOOLS_PATH}/win64/bin/moviLLD.exe")
    if(build_bitcode)
      # Assume these are in the environment until they can be shipped
      # with as part of the movicompile package.
      set(mv_tools_llvm_link "llvm-link.exe")
      set(mv_tools_llvm_ar "llvm-ar.exe")
    endif()
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
endif()

# Make list of all kernel descriptors
if(build_kernels)
  file(GLOB kernel_descrip_list RELATIVE "${kernel_descrip_dir}" CONFIGURE_DEPENDS "${kernel_descrip_dir}/*.txt")
  set(KERNELS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

#
# Process descriptors and build kernels
#
foreach(kernel_descrip IN LISTS kernel_descrip_list)
  ### PARSE DESCRIPTOR ###
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

  # Reading descrip file
  include("${kernel_descrip_dir}/${kernel_descrip}")

  if(NOT kernel_cpunum)
    message(SEND_ERROR "Missing kernel_cpunum in descriptor ${kernel_descrip}")
  elseif(NOT kernel_src)
    message(SEND_ERROR "Missing kernel_src in descriptor ${kernel_descrip}")
  endif()

  get_filename_component(kernel_name ${kernel_src} NAME_WE)
  if(kernel_entry STREQUAL "")
    set(kernel_entry "${kernel_name}")
  endif()

  # Collect names of all binaries for prebuild clean-up
  set(kernel_cpu "${kernel_cpunum}xx")
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")
  set(elf_file "${kernel_name}${kernel_cpu_suffix}${elf_suffix}")
  list(APPEND all_kernels "${elf_file}")
  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
    set(elf_file_lsu0_wo "${kernel_name}${kernel_cpu_suffix}_lsu0_wo${elf_suffix}")
    list(APPEND all_kernels "${elf_file_lsu0_wo}")
  endif()

  # We no longer want MTL MGMT kernel to be built and updated by default
  if(kernel_cpunum STREQUAL "3720" AND rt_kernel)
    continue()
  endif()
  # Skip kernel build if not enabled
  if((rt_kernel AND NOT ENABLE_MANAGEMENT_KERNEL_BUILD) OR
     (firmware_kernel AND NOT ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD) OR
     (NOT (rt_kernel OR firmware_kernel) AND NOT ENABLE_SHAVE_BINARIES_BUILD))
    continue()
  endif()

  if(kernel_cpunum STREQUAL "3720")
    list(APPEND define_symbols_list "USE_3720_INSTRUCTIONS")
    set(link_libraries_list ${link_libraries_list_VPU3720})
    if(rt_kernel)
      list(APPEND define_symbols_list "CONFIG_TARGET_SOC_3720")
    endif()
  elseif(kernel_cpunum STREQUAL "4000")
    list(APPEND define_symbols_list "USE_4000_INSTRUCTIONS")
    set(link_libraries_list ${link_libraries_list_VPU4000})
    if(rt_kernel)
      list(APPEND define_symbols_list "CONFIG_TARGET_SOC_4000")
    endif()
  endif()

  if(always_inline)
    list(APPEND define_symbols_list "CONFIG_ALWAYS_INLINE")
  endif()

  list(INSERT include_dirs_list 0 "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  list(INSERT define_symbols_list 0 "__act_shave__")

  ### RESOLVE FILE NAMES ###
  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")
  if (IS_ABSOLUTE "${kernel_src_file}")
    set(kernel_src_path "${kernel_src_file}")
  else()
    set(kernel_src_path "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_src_file}")
  endif()

  set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")
  set(dep_file "${kernel_src}${kernel_cpu_suffix}.d")
  set(elf_path "${target_binary_dir}/${elf_file}")
  set(asm_file "${kernel_name}${kernel_cpu_suffix}${asm_suffix}")
  set(asm_path "${target_asm_dir}/${asm_file}")
  set(bitcode_file "${kernel_src}${kernel_cpu_suffix}${bitcode_suffix}")
  set(linked_bitcode_file "${kernel_src}${kernel_cpu_suffix}${linked_bitcode_suffix}")
  set(linked_bitcode_path "${target_bitcode_dir}/${linked_bitcode_file}")
  set(asm_archive_file "${kernel_src}${kernel_cpu_suffix}${ar_suffix}")
  set(asm_archive_path "${target_asm_archive_dir}/${asm_archive_file}")
  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
    set(obj_file_lsu0_wo "${kernel_src}${kernel_cpu_suffix}_lsu0_wo${obj_suffix}")
    set(dep_file_lsu0_wo "${kernel_src}${kernel_cpu_suffix}_lsu0_wo.d")
    set(elf_path_lsu0_wo "${target_binary_dir}/${elf_file_lsu0_wo}")
    set(asm_file_lsu0_wo "${kernel_name}${kernel_cpu_suffix}_lsu0_wo${asm_suffix}")
    set(asm_path_lsu0_wo "${target_asm_dir}/${asm_file_lsu0_wo}")
    set(cppflags_list_lsu0_wo
      "-mllvm"
      "-shave-lsu-load-policy=use-only-lsu1"
      "-mllvm"
      "-shave-lsu-store-policy=prefer-lsu0"
    )
  endif()

  if(rt_kernel)
    # ACT management kernel must be built with Shave preemption checks disabled
    set(shave_preemption_opt "-mshave-preemption-checks=off")
    set(link_script_file "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_rt_kernel.ld")
  else()
      set(shave_preemption_opt "-mshave-preemption-checks=restore")
      list(APPEND shave_preemption_opt "-mshave-low-impact-preemption" )
      list(APPEND shave_preemption_opt "-mshave-preemption-max-loop-depth=1")
  endif()

  # List needed to create the final embedded description file
  list(TRANSFORM include_dirs_list PREPEND "-I")
  list(TRANSFORM define_symbols_list PREPEND "-D")

  set(compile_options
    "-mcpu=${kernel_cpu}"
    ${optimization_opts}
    ${cppflags_list}
    ${include_dirs_list}
    ${define_symbols_list}
    ${shave_preemption_opt})

  # Compile the kernel and output an assembly file
  # No output, no dependencies -- always regenerate
  add_custom_target("${asm_file}"
    COMMAND "${mv_tools_compile}" ${compile_options} -S "${kernel_src_path}" -o "${asm_path}"
    COMMENT "Generating ${asm_file}"
  )
  add_dependencies(act_shave_kernels_asm "${asm_file}")

  # Compile the kernel and output an object file
  add_custom_command(
    OUTPUT "${obj_file}"
    DEPENDS "${kernel_src_path}"
    DEPFILE "${dep_file}"
    COMMAND "${mv_tools_compile}" -MD ${compile_options} -c "${kernel_src_path}" -o "${obj_file}"
  )

  set(obj_file_list "${obj_file}")

  if(build_bitcode)
    add_custom_command(
      OUTPUT "${bitcode_file}"
      DEPENDS "${kernel_src_file}"
      COMMAND "${mv_tools_compile}" -MD ${compile_options} -c -emit-llvm "${kernel_src_path}" -o "${bitcode_file}"
    )
    set(bitcode_file_list "${bitcode_file}")
  endif()

  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
    add_custom_target("${asm_file_lsu0_wo}"
      COMMAND "${mv_tools_compile}" ${compile_options} ${cppflags_list_lsu0_wo} -S "${kernel_src_path}" -o "${asm_path_lsu0_wo}"
      COMMENT "Generating ${asm_file_lsu0_wo}"
    )
    add_dependencies(act_shave_kernels_asm "${asm_file_lsu0_wo}")

    add_custom_command(
      OUTPUT "${obj_file_lsu0_wo}"
      DEPENDS "${kernel_src_path}"
      DEPFILE "${dep_file_lsu0_wo}"
      COMMAND "${mv_tools_compile}" -MD ${compile_options} ${cppflags_list_lsu0_wo} -c "${kernel_src_path}" -o "${obj_file_lsu0_wo}"
    )

    set(obj_file_list_lsu0_wo "${obj_file_lsu0_wo}")
  endif()

  # Compile extra sources if specified in the descrip file
  if(extra_src_list)
    foreach(extra_src_file ${extra_src_list})
      get_filename_component(src_name ${extra_src_file} NAME_WE)
      get_filename_component(dir_name ${extra_src_file} DIRECTORY)

      # Discard full path, just keep name of the dir the file is in
      get_filename_component(dir_name ${dir_name} NAME_WE)

      # Some extra_src files have the same filename so the .o files must be in separate directories to avoid overwriting.
      set(obj_path "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}")
      file(MAKE_DIRECTORY "${obj_path}")
      set(base_name "${obj_path}/${src_name}.${kernel_cpu}")
      set(obj_file "${base_name}${obj_suffix}")
      set(dep_file "${base_name}.d")

      add_custom_command(
        OUTPUT "${obj_file}"
        DEPENDS "${kernel_src_path}"
        DEPFILE "${dep_file}"
        COMMAND "${mv_tools_compile}" -MD ${compile_options} -c "${extra_src_file}" -o "${obj_file}"
      )

      list(APPEND obj_file_list "${obj_file}")
      if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
        list(APPEND obj_file_list_lsu0_wo "${obj_file}")
      endif()

      if (build_bitcode)
        set(bitcode_file "${base_name}${bitcode_suffix}")
        add_custom_command(
          OUTPUT "${bitcode_file}"
          DEPENDS "${kernel_src_file}"
          DEPFILE "${dep_file}"
          COMMAND "${mv_tools_compile}" -MD ${compile_options} -c -emit-llvm "${extra_src_file}" -o "${bitcode_file}"
        )
        list(APPEND bitcode_file_list "${bitcode_file}")
      endif()
    endforeach()
  endif()

  # Compile asm files if specified in the descrip file
  if(asm_src_list)
    # Due to lack of depfile support
    # find all .inc files in source directories
    set(inc_list)
    set(ar_file_list)
    foreach(inc_dir IN LISTS asm_src_list kernel_src_dir)
      file(GLOB_RECURSE inc_files CONFIGURE_DEPENDS "${inc_dir}/*.inc")
      LIST(APPEND inc_list ${inc_files})
    endforeach()

    foreach(asm_src_file ${asm_src_list})
      get_filename_component(src_name ${asm_src_file} NAME_WE)
      get_filename_component(dir_name ${asm_src_file} DIRECTORY)

      # Discard full path, just keep name of the dir the file is in
      get_filename_component(dir_name ${dir_name} NAME_WE)

      # Some asm_src files have the same filename so the .o files must be in separate directories to avoid overwriting.
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}")
      set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/asm_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")

      add_custom_command(
        OUTPUT "${obj_file}"
        DEPENDS "${asm_src_file}" "${inc_list}"
        COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${asm_include_list} -c "${asm_src_file}" -o "${obj_file}"
      )
      list(APPEND obj_file_list "${obj_file}")
      list(APPEND ar_file_list "${obj_file}")

      if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
        list(APPEND obj_file_list_lsu0_wo "${obj_file}")
      endif()
    endforeach()

    if(build_bitcode)
      add_custom_command(
        OUTPUT "${asm_archive_path}"
        DEPENDS "${ar_file_list}"
        COMMAND "${mv_tools_llvm_ar}" --format=gnu Dqcs "${asm_archive_path}" ${ar_file_list}
      )
      add_custom_target("${asm_archive_file}" DEPENDS "${asm_archive_path}")
      add_dependencies(act_shave_kernels_build "${asm_archive_file}")
      list(APPEND kernels_to_build "${asm_archive_path}")
    endif()
  endif()

  set(link_options
    "--script" "${link_script_file}"
    "-entry" "${kernel_entry}"
    "--gc-sections"
    "--strip-debug"
    "--discard-all"
    "-zmax-page-size=16")

  # Link the sources, add entry point and windowed sections, then dump the elf file
  add_custom_command(
    OUTPUT "${elf_path}"
    DEPENDS ${obj_file_list} "${link_script_file}" ${link_libraries_list}
    COMMAND "${mv_tools_link}" ${link_options} ${obj_file_list} -EL ${link_libraries_list} --output "${elf_path}"
  )
  list(APPEND kernels_to_build "${elf_path}")
  add_custom_target("${elf_file}" DEPENDS "${elf_path}")
  add_dependencies(act_shave_kernels_build "${elf_file}")

  if(build_bitcode)
    add_custom_command(
      OUTPUT "${linked_bitcode_path}"
      DEPENDS ${bitcode_file_list}
      COMMAND "${mv_tools_llvm_link}" ${bitcode_file_list} -o "${linked_bitcode_path}"
    )
    add_custom_target("${linked_bitcode_file}" DEPENDS "${linked_bitcode_path}")
    add_dependencies(act_shave_kernels_build "${linked_bitcode_file}")
    list(APPEND kernels_to_build "${linked_bitcode_path}")
  endif()

  if(kernel_cpunum STREQUAL "3720" AND NOT rt_kernel) # MTL DDR split access workaround
    add_custom_command(
      OUTPUT "${elf_path_lsu0_wo}"
      DEPENDS ${obj_file_list_lsu0_wo} "${link_script_file}" ${link_libraries_list}
      COMMAND "${mv_tools_link}" ${link_options} ${obj_file_list_lsu0_wo} -EL ${link_libraries_list} --output "${elf_path_lsu0_wo}"
    )
    list(APPEND kernels_to_build "${elf_path_lsu0_wo}")
    add_custom_target("${elf_file_lsu0_wo}" DEPENDS "${elf_path_lsu0_wo}")
    add_dependencies(act_shave_kernels_build "${elf_file_lsu0_wo}")
  endif()

endforeach()

### UPDATE PREBUILT KERNELS ###
file(GLOB prebuild_kernels RELATIVE "${prebuild_binary_dir}" CONFIGURE_DEPENDS "${prebuild_binary_dir}/*")
if(build_kernels)
  if(NOT kernels_to_build)
    message(FATAL_ERROR "No kernels to build")
  endif()

  # Remove prebuild binaries if descriptor is removed
  set(kernels_to_remove ${prebuild_kernels})
  list(REMOVE_ITEM kernels_to_remove ${all_kernels})
  foreach(kernel IN LISTS kernels_to_remove)
    message(STATUS "No descriptor for prebuild kernel ${kernel}, removing.")
    file(REMOVE "${prebuild_binary_dir}/${kernel}")
  endforeach()

  # If building kernels update prebuilt kernels unconditionally to ensure the
  # prebuilt kernels are up to date even after git operations
  # No output or byproducts to avoid removing these during clean
  add_custom_target(update_prebuilt_binaries
    DEPENDS act_shave_kernels_build
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${kernels_to_build} "${prebuild_binary_dir}"
    COMMENT "Updating prebuilt kernels"
  )
else()
  # dummy target if not building kernels
  add_custom_target(update_prebuilt_binaries)
endif()

### LIBRARY TARGET ####
# Add file-level and target-level dependencies, we need this regenerated after
# kernels build and after any git actions
list(TRANSFORM prebuild_kernels PREPEND "${prebuild_binary_dir}/" OUTPUT_VARIABLE prebuild_bins)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -D KERNELS_BIN_DIR="${prebuild_binary_dir}"
                           -D GENERATED_FILE="generated_shave_binary_resources.cpp"
                           -D MAP_NAME="shaveBinaryResourcesMap"
                           -D EXTENSION="${elf_suffix}"
                           -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shave_binaries.cmake"
  DEPENDS act_shave_kernels_build update_prebuilt_binaries ${prebuild_bins} ${kernels_to_build}
)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_bitcode_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -D KERNELS_BIN_DIR="${prebuild_binary_dir}"
                           -D GENERATED_FILE="generated_shave_bitcode_resources.cpp"
                           -D MAP_NAME="shaveBitcodeResourcesMap"
                           -D EXTENSION="${linked_bitcode_suffix}"
                           -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shave_binaries.cmake"
  DEPENDS act_shave_kernels_build update_prebuilt_binaries ${prebuild_bins} ${kernels_to_build}
)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_asm_archive_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -D KERNELS_BIN_DIR="${prebuild_binary_dir}"
                           -D GENERATED_FILE="generated_shave_asm_archive_resources.cpp"
                           -D MAP_NAME="shaveAsmArchiveResourcesMap"
                           -D EXTENSION="${ar_suffix}"
                           -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shave_binaries.cmake"
  DEPENDS act_shave_kernels_build update_prebuilt_binaries ${prebuild_bins} ${kernels_to_build}
)

add_library(act_shave_kernels_lib OBJECT
    "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_bitcode_resources.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_asm_archive_resources.cpp"
)
# The library contains many large arrays, because of which it compiles slowly if the compiler optimisations are enabled.
target_compile_options(act_shave_kernels_lib PRIVATE -O0)
