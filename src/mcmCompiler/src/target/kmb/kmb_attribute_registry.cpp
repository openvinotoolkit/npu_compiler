//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// First register base attributes. //
#include    "src/base/attribute_registry.cpp"

// Kmb specific attributes. //

#include    "src/target/kmb/attribute_def/quantization_params.cpp"
#include    "src/target/kmb/attribute_def/barrier_definition.cpp"
#include    "src/target/kmb/attribute_def/barrier_deps.cpp"
#include    "src/target/kmb/attribute_def/dma_direction.cpp"
#include    "src/target/kmb/attribute_def/workloads.cpp"
#include    "src/target/kmb/attribute_def/ppe_task.cpp"
#include    "src/target/kmb/attribute_def/ppe_fixed_function.cpp"
#include    "src/target/kmb/attribute_def/custom_pwl_table.cpp"
