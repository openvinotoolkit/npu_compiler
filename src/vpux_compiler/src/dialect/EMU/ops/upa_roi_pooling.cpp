//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(ROIPoolingUPAOp op) {
    const auto inShapeFeatureMap = getShape(op.input());
    const auto inShapeCoord = getShape(op.coords());

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(op, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(op, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    const auto output_size = parseIntArrayAttr<int64_t>(op.output_size());
    if (output_size.size() != 2) {
        return errorAt(op, "Dimension of pooled size is expected to be equal to 2. Got {0}", output_size.size());
    }

    if (output_size[0] <= 0 || output_size[1] <= 0) {
        return errorAt(op, "Pooled size attributes pooled_h and pooled_w should be positive.");
    }

    return mlir::success();
}
