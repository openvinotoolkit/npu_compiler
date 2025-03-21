//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attr_interfaces.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/attributes.hpp.inc>

namespace vpux {
namespace VPUIP {

//
// SparsityCompressionAttr
//

VPUIP::SparsityCompressionAttr getSparsityCompressionAttr(mlir::Type type);
mlir::Type setSparsityCompressionAttr(mlir::Type type, VPUIP::SparsityCompressionAttr sparsityCompressionAttr);

VPUIP::SparsityCompressionAttr tileSparsityCompression(VPUIP::SparsityCompressionAttr sparsityCompression,
                                                       ShapeRef tileOffsets, ShapeRef tileShape);
mlir::Type tileTypeSparsityCompression(mlir::Type type, ShapeRef tileOffsets, ShapeRef tileShape);

}  // namespace VPUIP
}  // namespace vpux
