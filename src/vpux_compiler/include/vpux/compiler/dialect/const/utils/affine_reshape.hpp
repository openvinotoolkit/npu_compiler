//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux::Const {

std::optional<mlir::Type> inferElemTypeAffineReshape(ShapeRef inputShape, mlir::Type inputElementType,
                                                     const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                     ArrayRef<int64_t> shapeValue);

std::optional<DimsOrder> inferAffineReshapeOutputLayout(const DimArr& inPerm, mlir::ArrayAttr dimMapAttr);

}  // namespace vpux::Const
