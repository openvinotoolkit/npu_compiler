//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// PadWithZeroAttr::verify
//

mlir::LogicalResult vpux::Const::PadWithZeroAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         mlir::ArrayAttr padBefore, mlir::ArrayAttr padAfter) {
    if (padBefore == nullptr) {
        return printTo(emitError(), "Got NULL 'padBefore' in 'PadWithZeroAttr'");
    }
    if (padAfter == nullptr) {
        return printTo(emitError(), "Got NULL 'padAfter' in 'PadWithZeroAttr'");
    }

    if (padBefore.size() != padAfter.size()) {
        return printTo(emitError(), "Got non consistent 'padBefore' and 'padAfter' values in 'PadWithZeroAttr'");
    }

    for (const auto dimAttr : padBefore.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value in 'padBefore' for 'PadWithZeroAttr'");
        }
    }
    for (const auto dimAttr : padAfter.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value in 'padAfter' for 'PadWithZeroAttr'");
        }
    }

    return mlir::success();
}

//
// PadWithZeroAttr::print
//

void vpux::Const::PadWithZeroAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getPadBefore());
    printer << ", ";
    printer.printAttribute(getPadAfter());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::PadWithZeroAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr padBefore;
    if (mlir::failed(parser.parseAttribute(padBefore))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr padAfter;
    if (mlir::failed(parser.parseAttribute(padAfter))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::PadWithZeroAttr>(padBefore, padAfter);
}

//
// PadWithZeroAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::PadWithZeroAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto padBefore = parseIntArrayAttr<int64_t>(getPadBefore());
    const auto padAfter = parseIntArrayAttr<int64_t>(getPadAfter());
    return input.pad(ShapeRef(padBefore), ShapeRef(padAfter));
}

bool vpux::Const::PadWithZeroAttr::inferOutputSplat(bool, vpux::NDTypeInterface) {
    return false;
}

//
// PadWithZeroAttr::transform
//

Const::Content vpux::Const::PadWithZeroAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    output.fillWithZero();

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    const Byte elemSize = getElemTypeSize(input.getStorageElemType());

    const auto order = input.getType().getDimsOrder();
    const auto inShape = input.getType().getShape();
    const auto inMemShape = order.toMemoryOrder(inShape);

    const auto outShape = output.getType().getShape();
    const auto outMemShape = order.toMemoryOrder(outShape);

    const auto padBefore = Shape(parseIntArrayAttr<int64_t>(getPadBefore()));
    const auto memPadBefore = order.toMemoryOrder(padBefore);

    if (memPadBefore.size() == 1) {
        // Opitimized 1D case

        if (input.isSplat()) {
            const auto md0 = MemDim(0);
            const auto IN0 = inMemShape[md0];
            const auto off0 = memPadBefore[md0];

            loop_1d(LoopExecPolicy::Parallel, getContext(), IN0, [&](int64_t in0) {
                const auto out0 = in0 + off0;

                std::copy_n(inBuf.data(), checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + checked_cast<size_t>(out0 * elemSize.count()));
            });
        } else {
            std::copy_n(inBuf.data(), checked_cast<size_t>(input.getType().getNumElements() * elemSize.count()),
                        outBuf.data() + memPadBefore.front() * elemSize.count());
        }
    } else if (memPadBefore.size() == 2) {
        // Opitimized 2D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];

        const auto OUT1 = outMemShape[md1];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];

        loop_2d(LoopExecPolicy::Parallel, getContext(), IN0, IN1, [&](int64_t in0, int64_t in1) {
            const auto out0 = in0 + off0;
            const auto out1 = in1 + off1;

            const auto outRawInd = out1 + out0 * OUT1;
            const auto inRawInd = input.isSplat() ? 0 : in1 + in0 * IN1;

            std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                        checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
        });
    } else if (memPadBefore.size() == 3) {
        // Opitimized 3D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);
        const auto md2 = MemDim(2);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];
        const auto IN2 = inMemShape[md2];

        const auto OUT1 = outMemShape[md1];
        const auto OUT2 = outMemShape[md2];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];
        const auto off2 = memPadBefore[md2];

        loop_3d(LoopExecPolicy::Parallel, getContext(), IN0, IN1, IN2, [&](int64_t in0, int64_t in1, int64_t in2) {
            const auto out0 = in0 + off0;
            const auto out1 = in1 + off1;
            const auto out2 = in2 + off2;

            const auto outRawInd = out2 + out1 * OUT2 + out0 * OUT2 * OUT1;
            const auto inRawInd = input.isSplat() ? 0 : in2 + in1 * IN2 + in0 * IN2 * IN1;

            std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                        checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
        });
    } else if (memPadBefore.size() == 4) {
        // Opitimized 4D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);
        const auto md2 = MemDim(2);
        const auto md3 = MemDim(3);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];
        const auto IN2 = inMemShape[md2];
        const auto IN3 = inMemShape[md3];

        const auto OUT1 = outMemShape[md1];
        const auto OUT2 = outMemShape[md2];
        const auto OUT3 = outMemShape[md3];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];
        const auto off2 = memPadBefore[md2];
        const auto off3 = memPadBefore[md3];

        loop_4d(LoopExecPolicy::Parallel, getContext(), IN0, IN1, IN2, IN3,
                [&](int64_t in0, int64_t in1, int64_t in2, int64_t in3) {
                    const auto out0 = in0 + off0;
                    const auto out1 = in1 + off1;
                    const auto out2 = in2 + off2;
                    const auto out3 = in3 + off3;

                    const auto outRawInd = out3 + out2 * OUT3 + out1 * OUT3 * OUT2 + out0 * OUT3 * OUT2 * OUT1;
                    const auto inRawInd =
                            input.isSplat() ? 0 : in3 + in2 * IN3 + in1 * IN3 * IN2 + in0 * IN3 * IN2 * IN1;

                    std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                                checked_cast<size_t>(elemSize.count()),
                                outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
                });
    } else {
        // Generic case

        loop_1d(LoopExecPolicy::Parallel, getContext(), input.getType().getNumElements(), [&](int64_t inMemInd1D) {
            const auto inMemIndND = getMemIndexND(inMemInd1D, inMemShape);

            MemShape outMemIndND(inMemIndND.size());
            for (auto ind : irange(outMemIndND.size())) {
                const auto md = MemDim(ind);
                outMemIndND[md] = inMemIndND[md] + memPadBefore[md];
            }

            const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);

            const auto inMemRawInd = input.isSplat() ? 0 : checked_cast<size_t>(inMemInd1D * elemSize.count());
            VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

            const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
            VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

            std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + outMemRawInd);
        });
    }

    return output;
}

//
// PadWithZeroAttr::getStableHashValue
//

llvm::hash_code vpux::Const::PadWithZeroAttr::getStableHashValue() const {
    const auto padBefore = parseIntArrayAttr<int64_t>(getPadBefore());
    const auto padAfter = parseIntArrayAttr<int64_t>(getPadAfter());
    return llvm::hash_combine(getMnemonic(), llvm::hash_combine_range(padBefore.begin(), padBefore.end()),
                              llvm::hash_combine_range(padAfter.begin(), padAfter.end()));
}
