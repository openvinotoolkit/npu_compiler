//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/attributes.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/ELF/IR/attributes.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <mlir/IR/BuiltinOps.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp.inc>
