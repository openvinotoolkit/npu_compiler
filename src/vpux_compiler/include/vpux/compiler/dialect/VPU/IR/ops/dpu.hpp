//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution_fwd.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise_fwd.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image_fwd.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling_fwd.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized_fwd.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image_fwd.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/dpu.hpp.inc>
