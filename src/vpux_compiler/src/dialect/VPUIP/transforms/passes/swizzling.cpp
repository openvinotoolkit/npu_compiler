//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/IRMapping.h>

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SWIZZLING
#define GEN_PASS_DEF_SWIZZLING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// Swizzling
//

class Swizzling final : public VPUIP::impl::SwizzlingBase<Swizzling> {
public:
    using ValuesSet = mlir::DenseSet<mlir::Value>;

public:
    explicit Swizzling(const bool enableWeightSwizzling, const bool enableActivationSwizzling, Logger log)
            : _enableWeightSwizzling(enableWeightSwizzling), _enableActivationSwizzling(enableActivationSwizzling) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableWeightSwizzling;
    bool _enableActivationSwizzling;
    // Flags used for debug purpose and performance experiments
    bool _checkConstantSizeAlignment = false;
    bool _enableSwizzlingOfFusedConsts = false;

    struct DeviceInfo {
        VPU::ArchKind archKind;
        int64_t cmxSize;
        int64_t reservedCMXSize;
    };

    struct OpSwizzlingFlags {
        bool activationInput{false};
        bool activationOutput{false};
        bool weightInput{false};
    };

    struct OpsInfo {
        SmallVector<mlir::Operation*> opsToRemove;
        DenseMap<VPUIP::NCEClusterTaskOp, OpSwizzlingFlags> opsSwizzlingFlagsMap;
    };

    // Store information about NCEClusterTask operands which got swizzled
    void safeRunOnFunc() final;
    void activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, DeviceInfo& deviceInfo,
                                   OpsInfo& opsInfo);
    void constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst,
                                 DeviceInfo& deviceInfo);
    template <typename InAllocOp, typename OutAllocOp>
    void addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp, mlir::Type newType,
                                        VPU::ArchKind archKind);
    void updateConstantTypeForSwizzling(Const::DeclareOp decOp, mlir::Operation* cstLoadOp, int64_t swizzlingKey,
                                        DeviceInfo& deviceInfo);
    ValuesSet getSwizzledOperandsFromFlagsMap(VPUIP::NCEClusterTaskOp nceOp, OpsInfo& opsInfo);
    bool canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp, DeviceInfo& deviceInfo, OpsInfo& opsInfo,
                           AliasesInfo& aliasesInfo);
    bool canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp, DeviceInfo& deviceInfo, OpsInfo& opsInfo,
                              AliasesInfo& aliasesInfo);
    bool checkCMXUsage(VPUIP::NCEClusterTaskOp, const ValuesSet& newBufsToSwizzle, DeviceInfo& deviceInfo,
                       OpsInfo& opsInfo, AliasesInfo& aliasesInfo);
};

void adjustReturnTypesForInputChain(mlir::Value value, int64_t swizzlingKey, VPU::ArchKind archKind) {
    auto adjustReturnType = [&](mlir::Value value) {
        auto adjustedType = setSwizzlingKey(value.getType(), swizzlingKey, archKind);
        value.setType(adjustedType);
    };

    adjustReturnType(value);
    if (auto viewOp = mlir::dyn_cast_or_null<VPUIP::ViewOp>(value.getDefiningOp())) {
        // Update the source return type, subview in this case
        auto subView = viewOp.getViewSource();
        adjustReturnType(subView);
    }
}

VPUIP::DistributedBufferType getDistributedBufferTypeWithSwizzling(VPUIP::DistributedBufferType origDistType,
                                                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    const auto ctx = origDistType.getContext();
    const auto order = origDistType.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto layoutAttr =
            vpux::MemRefAttr::get(orderAttr, nullptr, /*allocSize=*/nullptr, {swizzlingSchemeAttr}, ctx);

    return VPUIP::DistributedBufferType::get(ctx, origDistType.getShape().raw(), origDistType.getElementType(),
                                             layoutAttr, origDistType.getMemSpace(), origDistType.getDistribution(),
                                             origDistType.getSparsityCompression());
}

bool isSizeAlignmentRequired(Const::DeclareOp decOp, VPU::ArchKind archKind,
                             VPUIP::DistributedBufferType distributedType = nullptr) {
    auto isAlignmentRequired = [&](NDTypeInterface type) {
        auto swizzlingSizeAlignment = vpux::getSizeAlignmentForSwizzling(archKind);
        auto totalSize = type.getTotalAllocSize().count();
        if (totalSize % swizzlingSizeAlignment == 0) {
            return false;
        }
        return true;
    };

    auto type = decOp.getType().cast<NDTypeInterface>();
    if (distributedType == nullptr) {
        return isAlignmentRequired(type);
    } else {
        const auto distributionAttr = distributedType.getDistribution();
        const auto numClusters = distributionAttr.getNumClusters().getInt();

        const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                          "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(),
                          numClusters);

        const auto perClusterOffsets = distributedType.getPerClusterMemoryShapeOffsets();
        VPUX_THROW_UNLESS(perClusterOffsets.size() == checked_cast<size_t>(numClusters),
                          "Number of offsets '{0}' and clusters '{1}' are mismatch", perClusterOffsets.size(),
                          numClusters);

        auto elemType = type.getElementType();

        for (auto i = 0; i < numClusters; i++) {
            mlir::Type newType = nullptr;
            if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                const auto newQType = tileScalesAndZP(qType, perClusterShapes[i], perClusterOffsets[i]);
                newType = type.changeShapeElemType(perClusterShapes[i], newQType);
            } else {
                newType = type.changeShape(perClusterShapes[i]);
            }

            newType = VPUIP::tileTypeSparsityCompression(newType, perClusterOffsets[i], perClusterShapes[i])
                              .cast<vpux::NDTypeInterface>();

            if (isAlignmentRequired(newType)) {
                return true;
            }
        }
    }
    return false;
}

mlir::LogicalResult Swizzling::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableWeightsSwizzling.hasValue()) {
        _enableWeightSwizzling = enableWeightsSwizzling.getValue();
    }
    if (enableActivationSwizzling.hasValue()) {
        _enableActivationSwizzling = enableActivationSwizzling.getValue();
    }

    return mlir::success();
}

// Check if for a given operation adding swizzling for provided buffers will not cause in increase
// of memory demand beyond CMX size
bool Swizzling::checkCMXUsage(VPUIP::NCEClusterTaskOp nceOp, const ValuesSet& newBufsToSwizzle, DeviceInfo& deviceInfo,
                              OpsInfo& opsInfo, AliasesInfo& aliasesInfo) {
    VPUX_THROW_WHEN(nceOp == nullptr, "Unsupported type {0} to check the memory with swizzling.", nceOp->getName());

    ValuesSet operands(nceOp->getOperands().begin(), nceOp->getOperands().end());

    SmallVector<std::pair<int64_t, int64_t>> buffSizeAndAlignment;

    auto registeredSwizzledOperands = getSwizzledOperandsFromFlagsMap(nceOp, opsInfo);
    ValuesSet rootBuffs;
    for (auto operand : operands) {
        bool bufferSwizzled = false;

        if (newBufsToSwizzle.find(operand) != newBufsToSwizzle.end()) {
            bufferSwizzled = true;
        }

        if (registeredSwizzledOperands.find(operand) != registeredSwizzledOperands.end()) {
            bufferSwizzled = true;
        }

        // Since 'weights', 'weights sparsity map', and 'weights table' could be fused into a single root buffer
        // check if they refer to the same buffer here to avoid redundant computations
        auto rootBuff = *aliasesInfo.getRoots(operand).begin();
        if (rootBuffs.find(rootBuff) != rootBuffs.end()) {
            continue;
        }
        rootBuffs.insert(rootBuff);

        auto rootBuffType = mlir::cast<NDTypeInterface>(rootBuff.getType());
        auto totalSize = rootBuffType.getTotalAllocSize().count();

        if (bufferSwizzled) {
            buffSizeAndAlignment.push_back(
                    std::make_pair(alignSizeForSwizzling(totalSize, getSizeAlignmentForSwizzling(deviceInfo.archKind)),
                                   vpux::getAddressAlignmentForSwizzling(vpux::SWIZZLING_KEY_5, deviceInfo.archKind)));
        } else {
            buffSizeAndAlignment.push_back(std::make_pair(totalSize, vpux::DEFAULT_CMX_ALIGNMENT));
        }
    }

    std::sort(buffSizeAndAlignment.begin(), buffSizeAndAlignment.end());

    // Because at this stage the order of allocation that will be used by FeasibleMemoryScheduler is not known,
    // perform the check on CMX usage on all permutations
    do {
        int64_t freeAddress = deviceInfo.reservedCMXSize;

        for (auto& buf : buffSizeAndAlignment) {
            auto start = freeAddress;
            if (start % buf.second) {
                start += buf.second - start % buf.second;
            }
            auto end = start + buf.first;
            freeAddress = end;
        }

        if (freeAddress > deviceInfo.cmxSize) {
            return false;
        }

    } while (std::next_permutation(buffSizeAndAlignment.begin(), buffSizeAndAlignment.end()));

    return true;
}

// 2 Major checks are required, the rest are just null checks
// 1. Are weights constant <- These buffers are swizzled as part of activation swizzling
//                            so avoid double swizzling here
// 2. isSizeAlignmentRequired <- This case will be enabled with E#48057
bool Swizzling::canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp, DeviceInfo& deviceInfo, OpsInfo& opsInfo,
                                  AliasesInfo& aliasesInfo) {
    auto isTiled = nceOp->getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto weights = nceOp.getWeights();
    auto weightsSM = nceOp.getWeightsSparsityMap();
    auto weightTable = nceOp.getWeightTable();

    _log.trace("Check if weights swizzling can be enabled for NCEOp '{0}'", nceOp->getLoc());

    // Swizzling for ELTWISE is handled with activation swizzling
    if (weights == nullptr || weightTable == nullptr || nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
        _log.nest().trace("Cannot swizzle weights because of missed weights", nceOp->getLoc());
        return false;
    }

    // WA for disabling swizzling for compressed conv layer
    // E#56431 will remove workaround for a long term fix
    if (auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>()) {
        _log.nest().trace("Cannot swizzle weights because this is compressed conv");
        return false;
    }

    // Swizzling of 5-d weights breaks accuracy.
    // Swizzling must be formulated for each sub-tensor in the batch individually.
    // For example, for a weight tensor with shape 2x1x16x3x3 swizzling must apply to two 1x16x3x3 tensors.
    // Individual buffers are not available at this stage of compilation.
    // They appear after BatchMatMulToMatMul pass.
    // The performance impact of swizzling in this scenario is negligible.
    // The decision was to omit the feature for 5-d shapes.
    if (getShape(weights).size() >= DimsGroups5D::Filter::numDims) {
        _log.nest().trace("Cannot swizzle weights because they have rank 5 or higher.");
        return false;
    }

    if (isFused) {
        if (!_enableSwizzlingOfFusedConsts) {
            // Even though support is in place enabling this caused schedule change
            // that had negative impact on performance in few cases
            // E#73720 will try to remove this check
            _log.nest().trace("Do not swizzle weights in case of fused consts");
            return false;
        }
        mlir::Operation* op = nullptr;
        auto decOp = vpux::ConstantFusing::getConstAndDma(weights, &op);
        if (decOp == nullptr) {
            _log.nest().trace("Cannot swizzle weights because of missed declare op");
            return false;
        }
        if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, deviceInfo.archKind)) {
            _log.nest().trace("Cannot swizzle weights. Size alignment required");
            return false;
        }
    } else if (isTiled) {
        auto checkDistributedContent = [&](mlir::Value value) {
            auto copyOp = value.getDefiningOp<VPUIP::CopyOp>();
            if (copyOp == nullptr) {
                return false;
            }
            auto decOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
            if (decOp == nullptr) {
                return false;
            }
            auto distributedBufferOp = copyOp.getOutputBuff().getDefiningOp<VPURT::AllocDistributed>();
            if (distributedBufferOp == nullptr) {
                return false;
            }
            auto distrType = value.getType().cast<VPUIP::DistributedBufferType>();
            if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, deviceInfo.archKind, distrType)) {
                return false;
            }
            return true;
        };

        auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>();

        auto bufferTilingOp = shapeCastOp ? shapeCastOp.getSource() : weights;

        if (!checkDistributedContent(bufferTilingOp)) {
            _log.nest().trace("Cannot swizzle weights, because weights do not satisfy distributed type requirements");
            return false;
        }

        if (!checkDistributedContent(weightTable)) {
            _log.nest().trace(
                    "Cannot swizzle weights, because weightTable do not satisfy distributed type requirements");
            return false;
        }

        if (weightsSM != nullptr) {
            if (!checkDistributedContent(weightsSM)) {
                _log.nest().trace(
                        "Cannot swizzle weights, because weightsSM do not satisfy distributed type requirements");
                return false;
            }
        }
    } else {
        auto checkCopyOfContentForSwizzling = [&](VPUIP::CopyOp copyOp) {
            if (copyOp == nullptr) {
                return false;
            }

            auto decOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
            if (decOp == nullptr) {
                return false;
            }

            if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, deviceInfo.archKind)) {
                return false;
            }
            return true;
        };

        auto copyOp = weights.getDefiningOp<VPUIP::CopyOp>();
        if (!checkCopyOfContentForSwizzling(copyOp)) {
            _log.nest().trace("Cannot swizzle weights, because weights do not satisfy distributed type requirements");
            return false;
        }

        auto wtCopyOp = weightTable.getDefiningOp<VPUIP::CopyOp>();
        if (!checkCopyOfContentForSwizzling(wtCopyOp)) {
            _log.nest().trace(
                    "Cannot swizzle weights, because weightTable do not satisfy distributed type requirements");
            return false;
        }

        if (weightsSM != nullptr) {
            auto weightsSMCopyOp = weightsSM.getDefiningOp<VPUIP::CopyOp>();
            if (!checkCopyOfContentForSwizzling(weightsSMCopyOp)) {
                _log.nest().trace(
                        "Cannot swizzle weights, because weightSM do not satisfy distributed type requirements");
                return false;
            }
        }
    }

    ValuesSet operands = {weights, weightTable};
    if (weightsSM != nullptr) {
        operands.insert(weightsSM);
    }

    // Check if adding constants swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(nceOp, operands, deviceInfo, opsInfo, aliasesInfo)) {
        _log.nest().trace("Do not enable weights swizzling because of increase in memory demand beyond CMX size");
        return false;
    }

    _log.nest().trace("NCEOp weights are eligible for swizzling");
    return true;
}

template <typename InAllocOp, typename OutAllocOp>
void Swizzling::addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp, mlir::Type newType,
                                               VPU::ArchKind archKind) {
    auto swizzlingSchemeAttr = getSwizzlingSchemeAttr(newType);
    auto addressAlignment = vpux::getAddressAlignmentForSwizzling(swizzlingSchemeAttr.getKey().getInt(), archKind);
    auto addressAlignmentAttr = getIntAttr(&getContext(), addressAlignment);

    builder.setInsertionPoint(inAllocOp);

    auto origLoc = inAllocOp->getLoc();
    auto newLoc = appendLoc(origLoc, "_alloc_swizzling");

    auto outAllocOp = builder.create<OutAllocOp>(newLoc, newType, addressAlignmentAttr, swizzlingSchemeAttr.getKey());

    inAllocOp->replaceAllUsesWith(outAllocOp);
    inAllocOp->erase();
}

bool isTypeSwizzled(mlir::Type type) {
    vpux::MemRefAttr memRefAttr;
    type = VPUIP::extractDataType(type);
    if (auto bufferMemRefType = type.dyn_cast<mlir::MemRefType>()) {
        memRefAttr = bufferMemRefType.getLayout().dyn_cast<vpux::MemRefAttr>();
    } else if (auto distBufferType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        memRefAttr = distBufferType.getLayout().dyn_cast<vpux::MemRefAttr>();
    }

    if (memRefAttr && memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>() != nullptr) {
        return true;
    }

    return false;
}

void Swizzling::updateConstantTypeForSwizzling(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp, int64_t swizzlingKey,
                                               DeviceInfo& deviceInfo) {
    VPUX_THROW_WHEN(cstOp == nullptr, "DeclareOp was not found");
    // On top of existing transformation a new transformation is added to the content attribute
    // of weight table const. The new transformation will swizzle the constant with swizzle key parameter
    _log.nest().trace("Constant for swizzling transformation'{0}'", cstOp->getLoc());

    auto cstType = cstOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (auto swizzlingSchemeAttr = vpux::getSwizzlingSchemeAttr(cstType)) {
        // Check if swizzling transformation is already attached, this can happen when constant is shared
        // between 2 or more NCEOps or when constants are fused
        return;
    }

    auto newCstType = vpux::setSwizzlingKey(cstType, swizzlingKey, deviceInfo.archKind);
    mlir::OpBuilder builder(cstOp);
    auto newCstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), newCstType, cstOp.getContentAttr());
    cstLoadOp->setOperand(0, newCstOp.getOutput());
    if (cstOp->getUses().empty()) {
        cstOp.erase();
    }
}

void Swizzling::constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst,
                                        DeviceInfo& deviceInfo) {
    if (cst == nullptr) {
        return;
    }

    auto isTiled = nceOp->getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(&getContext(), deviceInfo.archKind, SWIZZLING_KEY_5);

    auto shapeCastOp = cst.getDefiningOp<VPUIP::ShapeCastOp>();
    if (isFused || shapeCastOp) {
        adjustReturnTypesForInputChain(cst, SWIZZLING_KEY_5, deviceInfo.archKind);
    }

    VPUIP::CopyOp copyOp;
    mlir::Operation* op = nullptr;
    if (isFused) {
        std::ignore = vpux::ConstantFusing::getConstAndDma(cst, &op);
        copyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(op);
    } else {
        mlir::Value val = shapeCastOp != nullptr ? shapeCastOp.getSource() : cst;
        copyOp = val.getDefiningOp<VPUIP::CopyOp>();
    }
    assert(copyOp != nullptr);

    if (isTypeSwizzled(copyOp.getOutputBuff().getType())) {
        // In case of constant folding buffer might have already been swizzled
        return;
    }

    auto decOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
    auto copyOpOutput = copyOp.getOutputBuff();
    mlir::Type newType;

    if (isTiled) {
        auto distributedBuffer = copyOpOutput.getDefiningOp<VPURT::AllocDistributed>();

        _log.trace("Enable swizzling for distributed constant buffer of NCE task - '{0}'", nceOp->getLoc());

        updateConstantTypeForSwizzling(decOp, copyOp, SWIZZLING_KEY_5, deviceInfo);

        auto origType = distributedBuffer.getType().cast<VPUIP::DistributedBufferType>();

        // Create new DistributedBufferType which as part of layout will have swizzling set
        newType = getDistributedBufferTypeWithSwizzling(origType, swizzlingSchemeAttr);

        // Create new allocation with swizzling enabled and required alignment setting
        addSwizzlingAttributesToBuffer<VPURT::AllocDistributed, VPURT::AllocDistributed>(builder, distributedBuffer,
                                                                                         newType, deviceInfo.archKind);

    } else {
        auto allocOp = copyOp.getOutputBuff().getDefiningOp<mlir::memref::AllocOp>();
        VPUX_THROW_WHEN(allocOp == nullptr, "Allocation operation was not identified");

        _log.trace("Enable swizzling for constant buffer of NCE task - '{0}'", nceOp->getLoc());

        updateConstantTypeForSwizzling(decOp, copyOp.getOperation(), SWIZZLING_KEY_5, deviceInfo);

        auto origType = allocOp.getType().cast<vpux::NDTypeInterface>();
        newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                origType.getMemSpace(), StridesRef(), swizzlingSchemeAttr,
                                VPUIP::getSparsityCompressionAttr(origType));

        addSwizzlingAttributesToBuffer<mlir::memref::AllocOp, VPURT::Alloc>(builder, allocOp, newType,
                                                                            deviceInfo.archKind);
    }
    // Create new CopyOp with correct result type
    builder.setInsertionPointAfter(copyOp);
    auto tempCopyOp =
            builder.create<VPUIP::CopyOp>(copyOp.getLoc(), newType, copyOp.getInput(), copyOp.getOutputBuff());
    copyOp->replaceAllUsesWith(tempCopyOp);
    copyOp->erase();
}

mlir::DenseSet<mlir::Value> Swizzling::getSwizzledOperandsFromFlagsMap(VPUIP::NCEClusterTaskOp nceOp,
                                                                       OpsInfo& opsInfo) {
    ValuesSet swizzledOperands;
    auto swizzSettingsIt = opsInfo.opsSwizzlingFlagsMap.find(nceOp);
    if (swizzSettingsIt == opsInfo.opsSwizzlingFlagsMap.end()) {
        return swizzledOperands;
    }
    if (swizzSettingsIt->second.activationInput) {
        swizzledOperands.insert(nceOp.getInput());
        if (nceOp.getInputSparsityMap() != nullptr) {
            swizzledOperands.insert(nceOp.getInputSparsityMap());
        }
    }
    if (swizzSettingsIt->second.weightInput) {
        swizzledOperands.insert(nceOp.getWeights());
        swizzledOperands.insert(nceOp.getWeightTable());
        if (nceOp.getWeightsSparsityMap()) {
            swizzledOperands.insert(nceOp.getWeightsSparsityMap());
        }
    }
    if (swizzSettingsIt->second.activationOutput) {
        swizzledOperands.insert(nceOp.getOutputBuff());
        if (nceOp.getOutputSparsityMap() != nullptr) {
            swizzledOperands.insert(nceOp.getOutputSparsityMap());
        }
    }
    return swizzledOperands;
}

bool Swizzling::canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp, DeviceInfo& deviceInfo, OpsInfo& opsInfo,
                                     AliasesInfo& aliasesInfo) {
    mlir::Value nceDataResult = nceOp.getOutput();
    mlir::Value maybeNceSMResult = nullptr;

    _log.trace("Check if output swizzling can be enabled for NCEOp '{0}'", nceOp->getLoc());

    // Swizzling of 5-d activations breaks accuracy.
    // Swizzling must be formulated for each sub-tensor in the batch individually.
    // For example, for an activation tensor with shape 4x1x16x32x64 swizzling must apply to four 1x16x32x64 tensors.
    // Individual buffers are not available at this stage of compilation.
    // They appear after BatchMatMulToMatMul pass.
    // The performance impact of swizzling in this scenario is negligible.
    // The decision was to omit the feature for 5-d shapes.
    if (getShape(nceDataResult).size() >= DimsGroups5D::Filter::numDims) {
        _log.nest().trace("Cannot swizzle activation buffers because they have rank 5 or higher.");
        return false;
    }

    ValuesSet nceResults = {nceOp.getOutput()};
    ValuesSet outputBuffs = {nceOp.getOutputBuff()};
    if (auto outSM = nceOp.getOutputSparsityMap()) {
        maybeNceSMResult = outSM;
        nceResults.insert(outSM);
        outputBuffs.insert(nceOp.getOutputSparsityMapBuff());
    }

    VPUX_THROW_UNLESS(nceResults.size() == 1 || nceResults.size() == 2,
                      "NCEClusterTaskOp should have exact 1 or 2 output buffers, but got {0}", nceResults.size());

    mlir::DenseSet<VPUIP::NCEClusterTaskOp> userTasks;

    // Find DPU->DPU buffers that can be swizzled
    for (auto user : nceDataResult.getUsers()) {
        auto userNCETaskOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(user);
        if (userNCETaskOp == nullptr) {
            _log.nest().trace("Cannot swizzle activation buffers because one of consumers is not "
                              "VPUIP.NCEClusterTaskOp, user NCEOp - '{0}'",
                              user->getLoc());
            return false;
        }
        userTasks.insert(userNCETaskOp);

        // In theory this is impossible by design, but better to double check
        if (maybeNceSMResult != nullptr && userNCETaskOp.getInputSparsityMap() != maybeNceSMResult) {
            _log.nest().trace("Cannot swizzle activation buffers because sparsity map is not directly consumed by "
                              "VPUIP.NCEClusterTaskOp");
            return false;
        }

        if (userNCETaskOp.getInputStorageElementTable() != nullptr) {
            // E#99232: add support for SEP
            return false;
        }
    }

    for (auto outBuff : outputBuffs) {
        auto* bufOp = outBuff.getDefiningOp();

        if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(bufOp)) {
            _log.nest().trace("Cannot swizzle activation buffers because buffer '{0}' is not defined by allocation op",
                              outBuff);
            return false;
        }
    }

    // Check if adding activation swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(nceOp, outputBuffs, deviceInfo, opsInfo, aliasesInfo)) {
        _log.nest().trace("Do not enable activation swizzling because of increase in memory demand beyond CMX size");
        return false;
    }

    // Check if adding activation swizzling on output of this op will not increase memory demand
    // beyond CMX size of user operations
    for (auto userTask : userTasks) {
        if (!checkCMXUsage(userTask, nceResults, deviceInfo, opsInfo, aliasesInfo)) {
            _log.nest().trace("Do not enable activation swizzling because of increase in memory demand beyond CMX size "
                              "of user task - '{0}'",
                              userTask->getLoc());
            return false;
        }
    }

    // Set information about input swizzling for user tasks
    for (auto userNceOp : userTasks) {
        auto userSwizzSettingsIt = opsInfo.opsSwizzlingFlagsMap.find(userNceOp);
        VPUX_THROW_WHEN(userSwizzSettingsIt == opsInfo.opsSwizzlingFlagsMap.end(),
                        "Not found swizzling settings for given NCEOp - {0}", userNceOp->getLoc());
        userSwizzSettingsIt->second.activationInput = true;
    }

    _log.nest().trace("NCEOp is eligible for swizzling");
    return true;
}

void Swizzling::activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp,
                                          DeviceInfo& deviceInfo, OpsInfo& opsInfo) {
    SmallVector<mlir::Type> newClusterResultTypes;

    auto getResultIndex = [&](mlir::Value outputBuffer) {
        // data always has index 0. Profiling isn't enabled yet, so sparsityMap's index 1
        auto type = outputBuffer.getType().cast<vpux::NDTypeInterface>();
        return type.getElemTypeSize() == Bit(1) ? 1 : 0;
    };

    for (auto bufVal : {nceOp.getOutputBuff(), nceOp.getOutputSparsityMapBuff()}) {
        if (bufVal == nullptr) {
            continue;
        }

        mlir::Operation* sourceAllocOp = bufVal.getDefiningOp();

        if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(sourceAllocOp)) {
            _log.trace("Cannot swizzle output buffer of '{0}', since it's not memref.Alloc or VPURT.AllocDistrubted",
                       nceOp->getLoc());
            return;
        }

        auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(&getContext(), deviceInfo.archKind, SWIZZLING_KEY_5);
        auto addressAlignment =
                vpux::getAddressAlignmentForSwizzling(swizzlingSchemeAttr.getKey().getInt(), deviceInfo.archKind);
        auto addressAlignmentAttr = getIntAttr(&getContext(), addressAlignment);

        auto origType = (*sourceAllocOp->getResultTypes().begin()).cast<vpux::NDTypeInterface>();

        const auto outputIndex = getResultIndex(bufVal);

        _log.trace("Enable swizzling for {0}th output buffer of NCE task - '{1}'", outputIndex, nceOp->getLoc());

        builder.setInsertionPoint(sourceAllocOp);
        auto newLoc = appendLoc(sourceAllocOp->getLoc(), "_alloc_swizzling");

        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(sourceAllocOp)) {
            // Create new MemRefType which as part of layout will have swizzling set
            auto newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                         origType.getMemSpace(), StridesRef(), swizzlingSchemeAttr,
                                         VPUIP::getSparsityCompressionAttr(origType));

            // Create new allocation with swizzling enabled and required alignment setting
            auto newAlloc =
                    builder.create<VPURT::Alloc>(newLoc, newType, addressAlignmentAttr, swizzlingSchemeAttr.getKey());
            allocOp->replaceAllUsesWith(newAlloc);

            opsInfo.opsToRemove.push_back(allocOp.getOperation());

            nceOp.getResult(outputIndex).setType(newType);

            // Update type for other Ops which also used this new allocation as output buffer
            for (auto op : newAlloc->getResult(0).getUsers()) {
                if (op != nceOp && mlir::isa<VPUIP::NCEClusterTaskOp>(op)) {
                    auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op);
                    for (auto& userBufVal :
                         {userClusterTaskOp.getOutputBuff(), userClusterTaskOp.getOutputSparsityMapBuff()}) {
                        if (userBufVal == nullptr || userBufVal != newAlloc) {
                            continue;
                        }
                        const auto userOutputIndex = getResultIndex(userBufVal);
                        userClusterTaskOp.getResult(userOutputIndex).setType(newType);
                    }
                }
            }
        } else if (auto allocOp = mlir::dyn_cast<VPURT::AllocDistributed>(sourceAllocOp)) {
            auto distributedType = origType.dyn_cast<VPUIP::DistributedBufferType>();

            // Create new DistributedBufferType which as part of layout will have swizzling set
            auto newType = getDistributedBufferTypeWithSwizzling(distributedType, swizzlingSchemeAttr);

            // Create new allocation with swizzling enabled and required alignment setting
            auto newAlloc = builder.create<VPURT::AllocDistributed>(newLoc, newType, addressAlignmentAttr,
                                                                    swizzlingSchemeAttr.getKey());
            allocOp->replaceAllUsesWith(newAlloc);

            opsInfo.opsToRemove.push_back(allocOp.getOperation());

            nceOp.getResult(outputIndex).setType(newType);

            newClusterResultTypes.push_back(newType);
        }
    }

    opsInfo.opsToRemove.push_back(nceOp.getOperation());
}

//
// safeRunOnFunc
//

// TODO: #71565
void Swizzling::safeRunOnFunc() {
    auto& aliasesInfo = getAnalysis<AliasesInfo>();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    DeviceInfo deviceInfo;
    deviceInfo.archKind = VPU::getArch(module);
    deviceInfo.cmxSize = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN).size().count();
    deviceInfo.reservedCMXSize = deviceInfo.cmxSize - VPU::getTotalCMXSize(module).count();

    OpsInfo opsInfo;

    if (!_enableActivationSwizzling && !_enableWeightSwizzling) {
        _log.trace("Swizzling is disabled");
        return;
    }

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&func.getBody().front().front(), &builderLog);

    // Iterate IR twice. First to check what NCEOps can have constats swizzled,
    // the second time check which NCEOps can have their output swizzled.
    // Second iteration is separated because when determining if NCEOp output can be swizzled
    // information about const swizzling is used.
    // In the end HW requires to have matching swizzling setting on NCE operands which
    // are consumed/produced by same reader/writer:
    // activation reader:
    // - input
    // - input_sparisty_map
    // weights reader
    // - weights
    // - weights_table
    // - weights_sparisty_map
    // output writer:
    // - output
    // - output_sparisty_map
    func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        opsInfo.opsSwizzlingFlagsMap[nceOp] = OpSwizzlingFlags();
        if (_enableWeightSwizzling && canSwizzleWeights(nceOp, deviceInfo, opsInfo, aliasesInfo)) {
            opsInfo.opsSwizzlingFlagsMap[nceOp].weightInput = true;
        }
    });

    if (_enableActivationSwizzling) {
        func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            if (canSwizzleActivation(nceOp, deviceInfo, opsInfo, aliasesInfo)) {
                opsInfo.opsSwizzlingFlagsMap[nceOp].activationOutput = true;
            }
        });
    }

    for (auto& opsSwizzlingFlags : opsInfo.opsSwizzlingFlagsMap) {
        auto nceOp = opsSwizzlingFlags.first;
        auto swizzFlags = opsSwizzlingFlags.second;

        // In case of weights constant is used by multi users, and some of those users can be swizzled and others
        // cannot(theoretically only weight will be shared, wt will not, so we only need to check weights here, same
        // as weightsSparsityMap). It will casuse accuracy issue if this weights constant is swizzled due to weights
        // and wt should be swizzled at the same time.
        if (swizzFlags.weightInput) {
            auto nceWeights = nceOp.getWeights();
            for (auto* userOp : nceWeights.getUsers()) {
                if (auto nceTaskOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(*userOp)) {
                    if (!opsInfo.opsSwizzlingFlagsMap[nceTaskOp].weightInput) {
                        opsSwizzlingFlags.second.weightInput = false;
                    }
                }
            }
        }
    }

    // Check Eltwise operations as they require same swizzling attribute on both inputs
    // There might be cases where disabling swizzling for one of eltwise inputs
    // might require doing the same for other eltwise ops which depend on the same buffer.
    // Instead of analyzing such potential chain of eltwise siblings (which should be
    // unlikely to occur) below code runs in a loop until no eltwise related
    // disabling of swizzling was performed
    bool anyEltwiseBuffersUpdated;
    do {
        anyEltwiseBuffersUpdated = false;
        func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            if (nceOp.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
                return;
            }

            auto input1 = nceOp.getInput();
            auto input2 = nceOp.getWeights();

            auto input1NceTask = input1.getDefiningOp<VPUIP::NCEClusterTaskOp>();
            auto input2NceTask = input2.getDefiningOp<VPUIP::NCEClusterTaskOp>();

            bool input1SwizzlingFlag = false;
            auto input1SwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.end();
            if (input1NceTask) {
                input1SwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.find(input1NceTask);
                input1SwizzlingFlag = ((input1SwizzlingFlagItr != opsInfo.opsSwizzlingFlagsMap.end()) &&
                                       input1SwizzlingFlagItr->second.activationOutput);
            }

            bool input2SwizzlingFlag = false;
            auto input2SwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.end();
            if (input2NceTask) {
                input2SwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.find(input2NceTask);
                input2SwizzlingFlag = ((input2SwizzlingFlagItr != opsInfo.opsSwizzlingFlagsMap.end()) &&
                                       input2SwizzlingFlagItr->second.activationOutput);
            }

            bool outputSwizzlingFlag = false;
            auto outputSwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.end();
            bool inplaceEltwise = false;
            if (nceOp.getIsInplace().value_or(false)) {
                inplaceEltwise = true;
                outputSwizzlingFlagItr = opsInfo.opsSwizzlingFlagsMap.find(nceOp);
                outputSwizzlingFlag = ((outputSwizzlingFlagItr != opsInfo.opsSwizzlingFlagsMap.end()) &&
                                       outputSwizzlingFlagItr->second.activationOutput);
            }

            if (input1SwizzlingFlag != input2SwizzlingFlag ||
                (inplaceEltwise &&
                 (input1SwizzlingFlag != outputSwizzlingFlag || input2SwizzlingFlag != outputSwizzlingFlag))) {
                _log.trace("Mismatch of swizzling setting of eltwise inputs, eltwise op - '{0}'", nceOp->getLoc());
                if (input1SwizzlingFlag) {
                    input1SwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input1NceTask->getLoc());
                    anyEltwiseBuffersUpdated = true;
                } else if (input2SwizzlingFlag) {
                    input2SwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input2NceTask->getLoc());
                    anyEltwiseBuffersUpdated = true;
                } else if (outputSwizzlingFlag) {
                    outputSwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on eltwise - '{0}'", nceOp->getLoc());
                    anyEltwiseBuffersUpdated = true;
                }
            }
        });
    } while (anyEltwiseBuffersUpdated);

    // After identifying operations which can have swizzling applied
    // perform actual enabling. Code before only made necessary checks
    // where swizzling can be applied and in some cases due to limitations
    // (e.g. eltwise input swizzling mismatch) swizzling enabling was reverted.
    for (auto opSwizzlingFlag : opsInfo.opsSwizzlingFlagsMap) {
        auto nceOp = opSwizzlingFlag.first;
        auto flags = opSwizzlingFlag.second;
        if (flags.weightInput) {
            constantBufferSwizzling(builder, nceOp, nceOp.getWeights(), deviceInfo);
            constantBufferSwizzling(builder, nceOp, nceOp.getWeightsSparsityMap(), deviceInfo);
            constantBufferSwizzling(builder, nceOp, nceOp.getWeightTable(), deviceInfo);
        }

        if (flags.activationOutput) {
            activationBufferSwizzling(builder, nceOp, deviceInfo, opsInfo);
        }
    }

    for (auto opToRemove : llvm::make_early_inc_range(opsInfo.opsToRemove)) {
        if (opToRemove->use_empty()) {
            opToRemove->erase();
        }
    }
}

}  // namespace

//
// createSwizzlingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSwizzlingPass(const bool enableWeightSwizzling,
                                                             const bool enableActivationSwizzling, Logger log) {
    return std::make_unique<Swizzling>(enableWeightSwizzling, enableActivationSwizzling, log);
}
