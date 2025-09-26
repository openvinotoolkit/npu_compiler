//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/activation.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/arithmetic.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/bitwise.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/comparison.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/control_flow.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/convolution.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/data_movement.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/eltwise.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/data_type.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/image.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/logical.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/normalization.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/pooling.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/recurrent.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/reduce.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/shape_manipulation.cpp.inc>
#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops/specialized.cpp.inc>
