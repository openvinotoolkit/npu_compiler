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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPUIP::DeclareTensorOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type memory,
                                         VPUIP::MemoryLocation locale, uint64_t dataIndex) {
    build(builder, state, memory, locale, builder.getI64ArrayAttr(ArrayRef<int64_t>{0}),  // localeIndex
          dataIndex,
          nullptr,  // sparsityIndex
          nullptr,  // storageElementIndex
          nullptr,  // storageElementSize
          nullptr,  // leadingOffset
          nullptr,  // trailingOffset
          nullptr   // swizzlingKey
    );
}

void vpux::VPUIP::DeclareTensorOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type memory,
                                         VPUIP::MemoryLocation locale, uint32_t localeIndex, uint64_t dataIndex) {
    build(builder, state, memory, locale, builder.getI64ArrayAttr(ArrayRef<int64_t>(static_cast<int64_t>(localeIndex))),
          dataIndex,
          nullptr,  // sparsityIndex
          nullptr,  // storageElementIndex
          nullptr,  // storageElementSize
          nullptr,  // leadingOffset
          nullptr,  // trailingOffset
          nullptr   // swizzlingKey
    );
}

void vpux::VPUIP::DeclareTensorOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type memory,
                                         VPUIP::MemoryLocation locale, ArrayRef<int64_t> localeIndex,
                                         uint64_t dataIndex) {
    build(builder, state, memory, locale, getIntArrayAttr(builder, localeIndex), dataIndex,
          nullptr,  // sparsityIndex
          nullptr,  // storageElementIndex
          nullptr,  // storageElementSize
          nullptr,  // leadingOffset
          nullptr,  // trailingOffset
          nullptr   // swizzlingKey
    );
}

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareTensorOp op) {
    const auto locale = op.locale();

    // TODO: check localeIndex

    const auto memref = op.memory().getType().cast<mlir::MemRefType>();

    if (!isMemoryCompatible(locale, memref)) {
        return errorAt(op, "Locale '{0}' is not compatible with memory space '{1}'", locale, memref.getMemorySpace());
    }

    // TODO: check other offsets

    return mlir::success();
}

mlir::ParseResult vpux::VPUIP::DeclareTensorOp::parseLocaleIndex(mlir::OpAsmParser& parser,
                                                                 mlir::ArrayAttr& localeIndex) {
    SmallVector<int64_t> indicies;
    const auto parseElemCb = [&]() -> mlir::ParseResult {
        int64_t idx = 0;
        if (mlir::failed(parser.parseInteger(idx))) {
            return mlir::failure();
        }
        indicies.emplace_back(idx);
        return mlir::success();
    };

    if (mlir::failed(parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::OptionalSquare, parseElemCb))) {
        return mlir::failure();
    }

    localeIndex = parser.getBuilder().getI64ArrayAttr(indicies);
    return mlir::success();
}

void vpux::VPUIP::DeclareTensorOp::printLocaleIndex(mlir::OpAsmPrinter& printer, VPUIP::DeclareTensorOp&,
                                                    mlir::ArrayAttr localeIndex) {
    if (!localeIndex.empty()) {
        printer.printAttribute(localeIndex);
        printer << ' ';
    }
}
