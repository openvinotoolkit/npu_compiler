//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/attributes.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/attributes.cpp.inc>
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/enums.cpp.inc>

//
// Dialect hooks
//

void vpux::NPUReg60XX::NPUReg60XXDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/attributes.cpp.inc>
            >();
}
