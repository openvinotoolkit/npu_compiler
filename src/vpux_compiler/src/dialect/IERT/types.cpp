//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/types.hpp"

#include "vpux/compiler/dialect/IERT/dialect.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/IERT/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES
