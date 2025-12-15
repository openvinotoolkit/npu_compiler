//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.cpp.inc>
#include <vpux/compiler/NPU50XX/dialect/NPUReg50XX/enums.cpp.inc>

//
// Dialect hooks
//

void vpux::NPUReg50XX::NPUReg50XXDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.cpp.inc>
            >();
}
