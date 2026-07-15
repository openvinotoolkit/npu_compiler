//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/convert_lut_to_const.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

mlir::LogicalResult VPUIP::LUTConverterBase::matchAndRewrite(VPUIP::NCEClusterTaskOp nceClusterTask,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), nceClusterTask->getName(), nceClusterTask->getLoc());

    const auto LUTConst = [&]() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(&_netFunc.getBody().front().front());
        return createLookupTableConst(nceClusterTask, rewriter);
    }();

    const auto copyDst = createCopyDestination(nceClusterTask, LUTConst, rewriter);
    const auto lutNceInput = rewriter.create<VPUIP::CopyOp>(nceClusterTask->getLoc(), LUTConst, copyDst).getOutput();

    replaceWithConstInput(nceClusterTask, lutNceInput, rewriter);

    return mlir::success();
}

mlir::Value VPUIP::LUTConverterBase::createCopyDestination(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value LUTConst,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto input = nceClusterTask.getInput();
    const auto inputType = input.getType();

    return llvm::TypeSwitch<mlir::Type, mlir::Value>(inputType)
            .Case<VPUIP::DistributedBufferType>([&](VPUIP::DistributedBufferType distributedBufferTypeIn) {
                const auto distributedInfo = createDistributionInfoAttr(distributedBufferTypeIn, nceClusterTask);
                const auto distributedBufferType =
                        createDistributedBufferType(distributedInfo, nceClusterTask, LUTConst);

                auto alignment = vpux::getIntAttr(nceClusterTask.getContext(), VPU::SPRLUT_ALIGNMENT_REQUIREMENT);

                auto allocDistributed = rewriter.create<VPURT::AllocDistributed>(
                        nceClusterTask.getLoc(), distributedBufferType, alignment, nullptr);
                return allocDistributed->getResult(0);
            })
            .Case<mlir::MemRefType>([&](auto) {
                const auto constOutType = mlir::dyn_cast<mlir::MemRefType>(LUTConst.getType());
                VPUX_THROW_WHEN(constOutType == nullptr,
                                "{0}: sprLUT const output type is expected to be MemRefType, but got {1}",
                                getDebugName(), LUTConst.getType());
                const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(nceClusterTask.getContext(),
                                                                      stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
                const auto cmxMemType = vpux::getMemRefType(
                        ShapeRef(constOutType.getShape()), constOutType.getElementType(),
                        mlir::cast<vpux::NDTypeInterface>(constOutType).getDimsOrder(), memSpaceCMX);
                const auto allocOp = rewriter.create<mlir::memref::AllocOp>(nceClusterTask.getLoc(), cmxMemType);
                return allocOp->getResult(0);
            })
            .Default([&](mlir::Type inputType) {
                VPUX_THROW("{0}: `{1}` is not supported as an input type", getDebugName(), inputType);
                return mlir::Value{};
            });
}

VPU::DistributionInfoAttr VPUIP::LUTConverterBase::createDistributionInfoAttr(
        VPUIP::DistributedBufferType inputDistribType, VPUIP::NCEClusterTaskOp nceClusterTask) const {
    auto inputDistribInfo = inputDistribType.getDistribution();
    VPUX_THROW_WHEN(inputDistribInfo == nullptr, "{0}: inputDistribInfo == nullptr for the input type is not allowed",
                    getDebugName());
    const auto duplicatedDistrModeAttr =
            VPU::DistributionModeAttr::get(nceClusterTask.getContext(), VPU::DistributionMode::DUPLICATED);
    return VPU::DistributionInfoAttr::get(nceClusterTask.getContext(), duplicatedDistrModeAttr, nullptr, nullptr,
                                          nullptr, nullptr, inputDistribInfo.getNumClusters(), nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPUIP::DistributedBufferType VPUIP::LUTConverterBase::createDistributedBufferType(
        VPU::DistributionInfoAttr distributedInfo, VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value LUTConst) const {
    const auto memSpaceCMX =
            vpux::IndexedSymbolAttr::get(nceClusterTask.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    const auto ndTypeInterface = mlir::cast<vpux::NDTypeInterface>(LUTConst.getType());
    return VPUIP::DistributedBufferType::get(
            nceClusterTask.getContext(), ndTypeInterface.getShape().raw(), ndTypeInterface.getElementType(),
            mlir::dyn_cast<mlir::MemRefType>(LUTConst.getType()).getLayout(), memSpaceCMX, distributedInfo);
}
