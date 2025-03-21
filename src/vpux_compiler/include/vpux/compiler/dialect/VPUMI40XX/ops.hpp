//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <mlir/IR/BuiltinOps.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUMI40XX/ops.hpp.inc>
