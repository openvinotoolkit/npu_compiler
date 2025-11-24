//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

namespace {

template <typename T>
void barrierPrintProperties(mlir::MLIRContext* /*ctx*/, mlir::OpAsmPrinter& printer, const T& properties,
                            mlir::ArrayRef<llvm::StringRef> /*elidedProps*/) {
    const auto hasIsFinalBarrier = properties.isFinalBarrier != nullptr;
    const auto hasIsStartBarrier = properties.isStartBarrier != nullptr;
    const auto hasWlmPage = properties.wlmPage != nullptr;
    const auto hasBarrierIndex = properties.barrierIndex.has_value();
    if (!hasIsFinalBarrier && !hasIsStartBarrier && !hasWlmPage && !hasBarrierIndex) {
        return;
    }
    bool shouldPrintComma = false;
    printer << "<{";
    if (hasIsFinalBarrier) {
        printer << "isFinalBarrier";
        shouldPrintComma = true;
    }
    if (hasIsStartBarrier) {
        if (shouldPrintComma) {
            printer << ", ";
        }
        printer << "isStartBarrier";
        shouldPrintComma = true;
    }
    if (hasWlmPage) {
        if (shouldPrintComma) {
            printer << ", ";
        }
        printer << "wlmPage = " << properties.wlmPage;
        shouldPrintComma = true;
    }
    if (hasBarrierIndex) {
        if (shouldPrintComma) {
            printer << ", ";
        }
        printer << "barrierIndex = " << properties.barrierIndex;
    }
    printer << "}>";
}

template <typename T>
mlir::ParseResult barrierParseProperties(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    const auto parseEnd = [&]() {
        if (mlir::failed(parser.parseRBrace())) {
            return mlir::failure();
        }
        if (mlir::failed(parser.parseGreater())) {
            return mlir::failure();
        }
        return mlir::success();
    };

    auto& prop = result.getOrAddProperties<T>();
    if (mlir::failed(parser.parseOptionalLess())) {
        return mlir::success();
    }
    if (mlir::failed(parser.parseLBrace())) {
        return mlir::failure();
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("isFinalBarrier"))) {
        prop.setIsFinalBarrier(mlir::UnitAttr::get(parser.getContext()));
        if (mlir::failed(parser.parseOptionalComma())) {
            return parseEnd();
        }
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("isStartBarrier"))) {
        prop.setIsStartBarrier(mlir::UnitAttr::get(parser.getContext()));
        if (mlir::failed(parser.parseOptionalComma())) {
            return parseEnd();
        }
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("wlmPage"))) {
        if (mlir::failed(parser.parseEqual())) {
            return mlir::failure();
        }
        mlir::IntegerAttr intAttr;
        if (mlir::failed(parser.parseAttribute(intAttr))) {
            return mlir::failure();
        }
        prop.setWlmPage(intAttr);
        if (mlir::failed(parser.parseOptionalComma())) {
            return parseEnd();
        }
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("barrierIndex"))) {
        if (mlir::failed(parser.parseEqual())) {
            return mlir::failure();
        }
        int64_t value = 0;
        if (mlir::failed(parser.parseInteger(value))) {
            return mlir::failure();
        }
        prop.setBarrierIndex(value);
    }
    return parseEnd();
}

}  // namespace

//
// ConfigureBarrierOp
//

mlir::LogicalResult vpux::VPURT::ConfigureBarrierOp::verify() {
    if (!getIsFinalBarrier()) {
        return mlir::success();
    }
    auto barrier = getBarrier();
    auto findConsumerOp = [&]() {
        SmallVector<VPURT::TaskOp> consumerOps;
        for (const auto& user : barrier.getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(user);
            VPUX_THROW_WHEN(taskOp == nullptr, "VPURT.TaskOp is expected as user for barrier at '{0}'", getLoc());
            auto waitBarriers = taskOp.getWaitBarriers();
            auto iter = llvm::find(waitBarriers, barrier);
            if (iter != waitBarriers.end()) {
                consumerOps.push_back(taskOp);
            }
        }
        return consumerOps;
    };
    auto consumerOps = findConsumerOp();
    if (!consumerOps.empty()) {
        return errorAt(getLoc(), "Final barrier at '{0}' has consumer op '{1}'", getLoc(), consumerOps);
    }
    return mlir::success();
}

std::optional<int64_t> vpux::VPURT::ConfigureBarrierOp::getBarrierIndex() {
    return getProperties().getBarrierIndex();
}

void vpux::VPURT::ConfigureBarrierOp::setBarrierIndex(std::optional<int64_t> value) {
    getProperties().setBarrierIndex(value);
}

void VPURT::ConfigureBarrierOp::printProperties(mlir::MLIRContext* ctx, mlir::OpAsmPrinter& printer,
                                                const Properties& properties,
                                                mlir::ArrayRef<llvm::StringRef> elidedProps) {
    barrierPrintProperties<Properties>(ctx, printer, properties, elidedProps);
}

mlir::ParseResult VPURT::ConfigureBarrierOp::parseProperties(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    return barrierParseProperties<Properties>(parser, result);
}

//
// DeclareVirtualBarrierOp
//

std::optional<int64_t> vpux::VPURT::DeclareVirtualBarrierOp::getBarrierIndex() {
    return getProperties().getBarrierIndex();
}

void vpux::VPURT::DeclareVirtualBarrierOp::setBarrierIndex(std::optional<int64_t> value) {
    getProperties().setBarrierIndex(value);
}

void VPURT::DeclareVirtualBarrierOp::printProperties(mlir::MLIRContext* ctx, mlir::OpAsmPrinter& printer,
                                                     const Properties& properties,
                                                     mlir::ArrayRef<llvm::StringRef> elidedProps) {
    barrierPrintProperties<Properties>(ctx, printer, properties, elidedProps);
}

mlir::ParseResult VPURT::DeclareVirtualBarrierOp::parseProperties(mlir::OpAsmParser& parser,
                                                                  mlir::OperationState& result) {
    return barrierParseProperties<Properties>(parser, result);
}
