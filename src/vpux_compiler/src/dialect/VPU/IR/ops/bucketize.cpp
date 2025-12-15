//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::BucketizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::BucketizeOpAdaptor bucketize(operands, attrs, prop);
    if (mlir::failed(bucketize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(bucketize.getData().getType());
    const auto outType = inType.changeElemType(bucketize.getOutputType());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::BucketizeOp::verify() {
    const mlir::Type constInt32 = getSInt32Type(getContext());
    const mlir::Type constInt64 = getSInt64Type(getContext());

    if (!(getOutputType() == constInt32 || getOutputType() == constInt64)) {
        return errorAt(*this,
                       "Attribute output_type support only SI32 and SI64 type according to schema definition for this "
                       "attribute. Got {0} type",
                       getOutputType());
    }

    return mlir::success();
}
