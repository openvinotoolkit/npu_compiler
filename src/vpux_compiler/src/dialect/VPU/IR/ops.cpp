//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/activation.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/arithmetic.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/bitwise.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/comparison.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/control_flow.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/convolution.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/data_movement.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/data_type.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/dpu.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/eltwise.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/image.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/internal.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/logical.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/m2i.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/normalization.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/pooling.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/recurrent.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/reduce.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/shape_manipulation.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/specialized.cpp.inc>
