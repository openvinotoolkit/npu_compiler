
//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

using namespace vpux;
using namespace VPUMI40XX;

//
// ConfigureBarrierOp
//

void ConfigureBarrierOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                               VPURegMapped::IndexType index, int64_t id, int64_t next_same_id,
                               mlir::IntegerAttr producer_count, mlir::IntegerAttr consumer_count) {
    build(odsBuilder, odsState, index, mlir::ValueRange{}, checked_cast<uint8_t>(id), next_same_id,
          /*previousSameId*/ nullptr, producer_count, consumer_count);
    return;
}

void ConfigureBarrierOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                               VPURegMapped::IndexType index, int64_t id, int64_t next_same_id,
                               mlir::IntegerAttr producer_count, mlir::IntegerAttr consumer_count,
                               bool isFinalBarrier) {
    build(odsBuilder, odsState, index, mlir::ValueRange{}, checked_cast<uint8_t>(id), next_same_id,
          /*previousSameId*/ nullptr, producer_count, consumer_count, isFinalBarrier);
    return;
}

void ConfigureBarrierOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                               VPURegMapped::IndexType index, int64_t id, int64_t next_same_id,
                               mlir::IntegerAttr producer_count, mlir::IntegerAttr consumer_count, bool isFinalBarrier,
                               bool isStartBarrier) {
    build(odsBuilder, odsState, index, mlir::ValueRange{}, checked_cast<uint8_t>(id), next_same_id,
          /*previousSameId*/ nullptr, producer_count, consumer_count, isFinalBarrier, isStartBarrier);
    return;
}

mlir::LogicalResult ConfigureBarrierOp::verify() {
    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (VPU::getArch(getOperation()) == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    const auto id = getId();
    const auto noOfAvailableBarriers = VPUIP::getNumAvailableBarriers(getOperation());
    if (id >= noOfAvailableBarriers) {
        return errorAt(getLoc(), "Operation {0}: barrier id {1} value is higher than available barriers {2}",
                       getOperationName(), id, noOfAvailableBarriers);
    }

    return ::mlir::success();
}

//
// Dot Printer
//

DotNodeColor ConfigureBarrierOp::getNodeColor() {
    return DotNodeColor::GREEN;
}

bool ConfigureBarrierOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                         llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    os << head << "real_id :" << middle << static_cast<uint16_t>(getId()) << end;
    os << head << "next_same_id :" << middle << getNextSameId() << end;
    os << head << "producer_count :" << middle << static_cast<uint16_t>(getProducerCount().value_or(0)) << end;
    os << head << "consumer_count :" << middle << static_cast<uint16_t>(getConsumerCount().value_or(0)) << end;
    return true;
}

DOT::EdgeDir ConfigureBarrierOp::getEdgeDirection(mlir::Operation*) {
    // don't really care about the ranges
    return DOT::EdgeDir::EDGE_SKIP;
}
