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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(PerAxisTileUPAOp op) {
    const auto inShape = getShape(op.input());
    if (checked_cast<size_t>(op.axis()) > inShape.size()) {
        return errorAt(op, "Tile axis '{0}' is out of range [1,{1}]", op.axis(), inShape.size());
    }
    return mlir::success();
}

void vpux::VPUIP::PerAxisTileUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles) {
    build(odsBuilder, odsState, input, output, axis, tiles, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PerAxisTileUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::TileParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axis()));
    builder.add_tiles(checked_cast<uint32_t>(tiles()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_TileParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseTile(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPATile supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPATile supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_TileParams();
    const auto axis = getIntAttr(_ctx, params->axis());
    const auto tiles = getIntAttr(_ctx, params->tiles());

    return builder.create<VPUIP::PerAxisTileUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], axis, tiles);
}
