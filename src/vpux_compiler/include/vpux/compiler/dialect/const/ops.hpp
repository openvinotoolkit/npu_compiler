//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

//
// Generated
//

#include <vpux/compiler/dialect/const/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.hpp.inc>
