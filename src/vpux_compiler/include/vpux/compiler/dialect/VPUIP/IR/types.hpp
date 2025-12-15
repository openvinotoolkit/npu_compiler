//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
