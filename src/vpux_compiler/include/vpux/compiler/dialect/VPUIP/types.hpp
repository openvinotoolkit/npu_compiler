//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
