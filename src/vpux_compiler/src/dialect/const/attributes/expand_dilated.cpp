//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

//
// ExpandDilatedAttr::verify
//

mlir::LogicalResult vpux::Const::ExpandDilatedAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                           mlir::ArrayAttr dilations) {
    if (dilations == nullptr) {
        return printTo(emitError(), "Got NULL 'dilations' in 'ExpandDilatedAttr'");
    }

    for (const auto dimDilationAttr : dilations.getValue()) {
        if (!mlir::isa<mlir::IntegerAttr>(dimDilationAttr)) {
            return printTo(emitError(), "Got non-integer value in 'dilations' for 'ExpandDilatedAttr'");
        }
        if (mlir::cast<mlir::IntegerAttr>(dimDilationAttr).getInt() <= 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'shape' for 'ExpandDilatedAttr'",
                           dimDilationAttr);
        }
    }

    return mlir::success();
}

//
// ExpandDilatedAttr::print
//

void vpux::Const::ExpandDilatedAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getDilations());
    printer << ">";
}

//
// ExpandDilatedAttr::parse
//

mlir::Attribute vpux::Const::ExpandDilatedAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr dilations;
    if (mlir::failed(parser.parseAttribute(dilations))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::ExpandDilatedAttr>(dilations);
}

//
// ExpandDilatedAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ExpandDilatedAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto dilations = parseIntArrayAttr<int64_t>(getDilations());
    auto tensor = mlir::cast<vpux::NDTypeInterface>(input);
    return getDilatedType(tensor, ShapeRef(dilations));
}

bool vpux::Const::ExpandDilatedAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    return false;
}

//
// ExpandDilatedAttr::transform
//

Const::Content vpux::Const::ExpandDilatedAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    output.fillWithZero();

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    const Byte elemSize = getElemTypeSize(input.getStorageElemType());

    const auto inShape = input.getType().getShape();
    const auto outShape = output.getType().getShape();

    const auto dilations = parseIntArrayAttr<int64_t>(getDilations());

    const auto OC = inShape[vpux::Dims4D::Filter::OC];
    const auto IC = inShape[vpux::Dims4D::Filter::IC];
    const auto KY = inShape[vpux::Dims4D::Filter::KY];
    const auto KX = inShape[vpux::Dims4D::Filter::KX];
    const auto dKY = outShape[vpux::Dims4D::Filter::KY];
    const auto dKX = outShape[vpux::Dims4D::Filter::KX];

    loop_4d(LoopExecPolicy::Parallel, input.getStorageElemType().getContext(), OC, IC, KY, KX,
            [&, inputIsSplat = input.isSplat()](int64_t oc, int64_t ic, int64_t ky, int64_t kx) {
                const auto dky = ky + (dilations[0] - 1) * ky;
                const auto dkx = kx + (dilations[1] - 1) * kx;

                const auto outRawInd = dkx + dky * dKX + ic * dKX * dKY + oc * dKX * dKY * IC;
                const auto inRawInd = kx + ky * KX + ic * KX * KY + oc * KX * KY * IC;

                // Note: in case of splats, there's no guarantee on how many
                // elements data has, so always use first element.
                const auto inBufBegin =
                        inputIsSplat ? inBuf.data() : inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count());

                // Note: this copies *single* element in a loop_4d...
                std::copy_n(inBufBegin, checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
            });

    return output;
}

//
// ExpandDilatedAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ExpandDilatedAttr::getStableHashValue() const {
    const auto dilations = parseIntArrayAttr<int64_t>(getDilations());
    return llvm::hash_combine(getMnemonic(), llvm::hash_combine_range(dilations.begin(), dilations.end()));
}
