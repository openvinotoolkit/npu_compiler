//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//
// The NPU Compiler version
// This version is exposed via L0 API and reported as (read-only) plugin property NPU_COMPILER_VERSION
//
#define NPU_COMPILER_VERSION_MAJOR 7
#define NPU_COMPILER_VERSION_MINOR 26

/*

Change Log:
-----------
NPU Compiler 7.26.0
  - UD44

NPU Compiler 7.25.0
  - Take NPU_QDQ_OPTIMIZATION_AGGRESSIVE property value to enable/disable aggressive qdq optimization.
  - UD38

NPU Compiler 7.23.0
  - UD38

NPU Compiler 7.22.0
  - UD32

NPU Compiler 7.21.0
  - UD28

NPU Compiler 7.20.1
  - Remove obsolete DPU_GROUPS config entry (NPU_DPU_GROUPS property)

NPU Compiler 7.20.0
  - Take NPU_QDQ_OPTIMIZATION property value to enable/disable adaptive stripping
  - Restricted public compiler options from NPU_COMPILATION_MODE_PARAMS to the following: optimization-level,
performance-hint-override, compute-layers-with-higher-precision, enable-activation-sparsity, enable-weights-sparsity,
enable-se-ptrs-operations, enable-wd-blockarg-input, split-bilinear-into-H-and-W, output-pipelining,
enable-output-ensurance, accumulate-matmul-with-dpu, fuse-fq-and-mul-with-non-const-input, workload-management-mode

NPU Compiler 7.4.0
  - Add option BATCH_COMPILER_MODE_SETTINGS and turn batch-compile-mode on 'debatch' by default

NPU Compiler 7.3.0
  - Remove model-hash option from NPU_COMPILATION_MODE_PARAMS

NPU Compiler 7.2.0
  - Add NF4 datatype support
  - Decouple NPU Compiler version from VCL version

VPUXCompilerL0 7.1.0
  - Add NPU_COMPILER_DYNAMIC_QUANTIZATION property

VPUXCompilerL0 7.0.0:
  - [VCL] Add vclGetVersion to show current vcl api version
  - [VCL] Add vcl_device_desc_t with members, deviceID, revision, tileCount
  - [VCL] Update vclCompilerCreate to use new structure
  - [VCL] Update default test platform to 4000

VPUXCompilerL0 6.3.0:
  - Bugfixes

VPUXCompilerL0 6.2.0:
  - Add support for NPU_DEFER_WEIGHTS_LOAD property

VPUXCompilerL0 6.1.0:
  - [VCL] Add vclAllocatedExecutableCreate to compile a network allocating blob storage via given allocator

VPUXCompilerL0 6.0.0:
  - [VCL] Add new data structure vcl_query_desc_t
  - [VCL] Change vclQueryNetworkCreate to use a vcl_query_desc_t instead of uint8_t* modelIR and uint64_t modelIRSize

VPUXCompilerL0 5.10.0:
  - Model hash is now automatically passed to the compiler to bypass multi-cluster assignment pass.

VPUXCompilerL0 5.9.0:
  - The I/O metadata will be identified using indices instead of names when the plugin version is new enough.

VPUXCompilerL0 5.8.0:
  - [VCL] Remove vpux_driver_compiler target and vpux_driver_compiler.h

VPUXCompilerL0 5.7.0:
  - Add support for optimization-level and performance-hint-override options of COMPILATION_MODE_PARAMS config

VPUXCompilerL0 5.6.2:
  - [VCL] Assign fixed value to vcl_platform_t

VPUXCompilerL0 5.6.1:
  - [VCL] Add npu_driver_compiler target and npu_driver_compiler.h

VPUXCompilerL0 5.6.0:
  - Add support for EXECUTION_MODE_HINT property.

VPUXCompilerL0 5.5.0:
  - Add support for NPU_BATCH_MODE option.

VPUXCompilerL0 5.4.1:
  - Add NPU_TILES to replace NPU_DPU_GROUPS.

VPUXCompilerL0 5.4.0:
  - Add support for INFERENCE_PRECISION_HINT compiler options.


VPUXCompilerL0 5.3.0:
  - Add support for NPU_MAX_TILES and NPU_STEPPING compiler options.

See src/vpux_driver_compiler/CHANGES.txt for earlier version history.

*/
