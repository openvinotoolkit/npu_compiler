//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/utils/core/custom_float.hpp"

using namespace vpux;

// This function is a recursive helper implementation of getConstAndDma
// It keeps on parsing the parent op and looks for the DeclareOp
// Once found stores the Op and returns the delcare Op
Const::DeclareOp getConstAndDmaRecImpl(mlir::BlockArgument arg, mlir::async::ExecuteOp execParentOp,
                                       mlir::Operation** constOp) {
    if (arg == nullptr || execParentOp == nullptr) {
        return nullptr;
    }

    // Adjust the index by adding dependencies size
    auto dependenciesSize = execParentOp.getDependencies().size();
    auto indexOfFusedConstant = arg.getArgNumber() + static_cast<int32_t>(dependenciesSize);

    // GoTo parent of the arg
    auto tempExecOp = execParentOp->getOperand(indexOfFusedConstant).getDefiningOp<mlir::async::ExecuteOp>();
    auto* tempBodyBlock = tempExecOp.getBody();
    for (auto& op : tempBodyBlock->getOperations()) {
        if (!mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(op)) {
            continue;
        }

        auto type = op.getResult(0).getType();
        if (auto ndType = type.cast<vpux::NDTypeInterface>()) {
            // For constant fusion this should always be U8 or F16
            if (!ndType.getElementType().isUnsignedInteger(8) && !ndType.getElementType().isF16()) {
                continue;
            }

            auto cstValue = mlir::cast<VPUIP::LayerOpInterface>(op).getInputs()[0];

            if (auto constDeclareOp = cstValue.getDefiningOp<Const::DeclareOp>()) {
                *constOp = &op;
                return constDeclareOp;
            }

            // Op is produced by other operation. By checking other users of this buffer
            // identify the one with const as input which would be the initial op loading searched constant
            for (auto user : cstValue.getUsers()) {
                bool isUserDistributed = vpux::VPUIP::hasDistributedOperand(user);
                if (isUserDistributed) {
                    auto newDecOp = user->getOperand(0).getDefiningOp<Const::DeclareOp>();
                    if (newDecOp != nullptr) {
                        *constOp = user;
                        return newDecOp;
                    }
                }
                if (!isUserDistributed && mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(user)) {
                    if (auto newDecOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(user)
                                                .getInputs()[0]
                                                .getDefiningOp<Const::DeclareOp>()) {
                        *constOp = user;
                        return newDecOp;
                    }
                }
            }

            // Op wrapped in async.execute has input but not found in this block
            // continue traversing by checking producer/parent of this argument
            arg = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].dyn_cast<mlir::BlockArgument>();
            execParentOp = op.getParentOfType<mlir::async::ExecuteOp>();
            return getConstAndDmaRecImpl(arg, execParentOp, constOp);
        }
    }
    return nullptr;
}

// Get the underlying Declare and Copy Op for the constant passed
// If not found on the first level recursively parse the parents of the Op until a DeclareOp is found
Const::DeclareOp ConstantFusing::getConstAndDma(mlir::Value constant, mlir::Operation** constOp) {
    Const::DeclareOp constDeclareOp = nullptr;
    VPUIP::ViewOp viewOp = nullptr;

    if (constant == nullptr) {
        return nullptr;
    }

    viewOp = constant.getDefiningOp<VPUIP::ViewOp>();
    VPUX_THROW_UNLESS(viewOp != nullptr, "Constant found without a ViewOp");

    auto subViewOp = viewOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");
    mlir::Value source = subViewOp.getSource();

    if (mlir::BlockArgument arg = source.dyn_cast<mlir::BlockArgument>()) {
        // Op wrapped in async.execute has input continue traversing by checking producer of this argument
        auto execParentOp = subViewOp->getParentOfType<mlir::async::ExecuteOp>();
        return getConstAndDmaRecImpl(arg, execParentOp, constOp);
    }

    if (auto declareBuffer = source.getDefiningOp<VPURT::DeclareBufferOp>()) {
        for (auto user : declareBuffer->getUsers()) {
            bool isUserDistributed = vpux::VPUIP::hasDistributedOperand(user);
            if (isUserDistributed) {
                *constOp = user;
                constDeclareOp = user->getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto allocDistributed = source.getDefiningOp<VPURT::AllocDistributed>()) {
        for (auto user : allocDistributed->getUsers()) {
            bool isDistributed = vpux::VPUIP::hasDistributedOperand(user);
            if (isDistributed) {
                *constOp = user;
                constDeclareOp = user->getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto* op = source.getDefiningOp()) {
        if (auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op)) {
            constDeclareOp = layerOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();
            while (constDeclareOp == nullptr) {
                op = layerOp.getInputs()[0].getDefiningOp();
                VPUX_THROW_UNLESS(op != nullptr, "Next CopyOp or NNDMAOp as source operation expected");
                layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
                if (layerOp == nullptr) {
                    break;
                }
                constDeclareOp = layerOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();
            }
            *constOp = op;
        }
    }

    return constDeclareOp;
}

int32_t ConstantFusing::getOffsetForConstant(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant) {
    int32_t offset = 0;
    VPUIP::ViewOp viewOp = nullptr;
    if (constant == nullptr) {
        return offset;
    }

    auto arg = constant.dyn_cast<mlir::BlockArgument>();
    if (arg != nullptr) {
        auto execParentOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        viewOp = execParentOp->getOperand(arg.getArgNumber()).getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Tiled Constant found without a ViewOp");
    } else {
        viewOp = constant.getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Getting Offset: Constant found without a ViewOp");
    }

    auto subViewOp = viewOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");

    auto offsets = subViewOp.getStaticOffsets();
    return parseIntArrayAttr<int32_t>(offsets).back();
}

VPUIP::DistributedBufferType ConstantFusing::getDistributedBufferType(VPUIP::DistributedBufferType origDistType,
                                                                      Const::DeclareOp declOp,
                                                                      mlir::PatternRewriter& rewriter) {
    auto typeInterface = declOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto ctx = typeInterface.getContext();
    const auto order = typeInterface.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto strides = typeInterface.getStrides();
    const Bit elemSize = typeInterface.getElemTypeSize();

    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));

    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layoutAttr = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                                  /*allocSize=*/nullptr, ctx);

    vpux::IndexedSymbolAttr memKindAttr =
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Create updated distributedTensorAttr, remove alignment as the fused buffer is a flat buffer
    auto origDistributionInfoAttr = origDistType.getDistribution();

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistributionInfoAttr)) {
        VPUX_THROW_WHEN(origDistributionInfoAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                        "DistributedBuffer for fused constant has mode different from DUPLICATED, type = {0}",
                        origDistType);

        auto newDistribution =
                VPU::getNonOverlappedDistributedAttr(typeInterface.getShape(), origDistributionInfoAttr.getMode(),
                                                     nullptr, origDistributionInfoAttr.getNumClusters(), nullptr,
                                                     origDistributionInfoAttr.getUniformDistributedSegments(), ctx);

        return VPUIP::DistributedBufferType::get(ctx, typeInterface.getShape().raw(), typeInterface.getElementType(),
                                                 layoutAttr, memKindAttr, newDistribution);
    }

    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(
            ctx, origDistributionInfoAttr.getMode(), origDistributionInfoAttr.getNumTiles(), nullptr, nullptr, nullptr,
            origDistributionInfoAttr.getNumClusters(), nullptr,
            origDistributionInfoAttr.getUniformDistributedSegments(), origDistributionInfoAttr.getComputeShapes(),
            origDistributionInfoAttr.getComputeOffsets(), origDistributionInfoAttr.getMemoryShapes(),
            origDistributionInfoAttr.getMemoryOffsets(), origDistributionInfoAttr.getEqualMemoryAndComputeView());

    return VPUIP::DistributedBufferType::get(ctx, typeInterface.getShape().raw(), typeInterface.getElementType(),
                                             layoutAttr, memKindAttr, distributedTensorAttr);
}

void ConstantFusing::getCopyAndDeclareOpForFusion(mlir::Value nceOperand, VPUIP::CopyOp& copyOp,
                                                  Const::DeclareOp& declareOp,
                                                  VPURT::AllocDistributed& foundAllocDistributed) {
    if (nceOperand == nullptr) {
        return;
    }
    // Don't fuse constants defined in ShapeCast for now
    // Also don't fuse constants that do not come from a CopyOp
    copyOp = nceOperand.getDefiningOp<VPUIP::CopyOp>();
    if (nceOperand.getDefiningOp<VPUIP::ShapeCastOp>() != nullptr || copyOp == nullptr) {
        return;
    }
    auto constantTypeDistributed = nceOperand.getType().dyn_cast<VPUIP::DistributedBufferType>();
    // Op is Tiled
    if (constantTypeDistributed != nullptr) {
        // Only Fuse if the constants are broadcasted/duplicated
        if (VPU::isDuplicated(constantTypeDistributed.getDistribution())) {
            foundAllocDistributed = copyOp.getOutputBuff().getDefiningOp<VPURT::AllocDistributed>();
            declareOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
        }
    } else {
        // If the constant isn't a block arg the parent Op is not tiled so just get the declare Op
        if (copyOp != nullptr) {
            declareOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();

            while (declareOp == nullptr) {
                copyOp = copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
                // If this is the case then the constant is not spilled, To be handled with E#45105
                if (copyOp == nullptr) {
                    // Return nullptr, will skip fusion for this layer
                    break;
                }

                declareOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
            }
        }
    }
}
