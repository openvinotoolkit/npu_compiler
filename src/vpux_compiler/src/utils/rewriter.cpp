//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/SmallPtrSet.h>

using namespace vpux;

//
// updateFunctionSignature
//

mlir::LogicalResult vpux::updateFunctionSignature(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                                  ArrayRef<mlir::Type> newResultTypes, Logger log) {
    const auto origFuncType = funcOp.getType();

    if (newArgTypes.size() != origFuncType.getNumInputs()) {
        log.trace("New inputs size '{0}' doesn't match original prototype", newArgTypes.size());
        return mlir::failure();
    }
    if (newResultTypes.size() != origFuncType.getNumResults()) {
        log.trace("New results size '{0}' doesn't match original prototype", newResultTypes.size());
        return mlir::failure();
    }

    const auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);

    if (newFuncType == origFuncType) {
        log.trace("Nothing to change");
        return mlir::success();
    }

    log.trace("Update Function signature : '{0}' -> '{1}'", origFuncType, newFuncType);
    funcOp.setType(newFuncType);

    return mlir::success();
}

namespace {

mlir::MemRefType tensorToBuffer(mlir::RankedTensorType tensorType) {
    const auto type = tensorType.cast<vpux::NDTypeInterface>();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return getMemRefType(shape, elemType, order, memSpace);
};

VPUIP::DistributedBufferType distributedTensorToBuffer(VPU::DistributedTensorType type) {
    return VPUIP::DistributedBufferType::get(type.getContext(), type.getShape().raw(), type.getElementType(),
                                             type.getOrder(), type.getMemSpace(), type.getDistribution());
}

mlir::Type bufferizeTensor(mlir::Type tensorType) {
    if (tensorType == nullptr) {
        return nullptr;
    }
    if (auto distributedType = tensorType.dyn_cast<VPU::DistributedTensorType>()) {
        return distributedTensorToBuffer(distributedType);
    } else if (auto rankedType = tensorType.dyn_cast<mlir::RankedTensorType>()) {
        return tensorToBuffer(rankedType);
    }
    VPUX_THROW("Unsupported type for bufferization '{0}'", tensorType);
}

}  // namespace

//
// convertFunc
//

mlir::LogicalResult vpux::convertFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                      ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.sym_name());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.sym_name());
        return mlir::failure();
    }

    if (updateFunctionSignature(funcOp, newArgTypes, newResultTypes, log).failed()) {
        return mlir::failure();
    }

    //
    // Convert arguments
    //

    log.trace("Convert arguments");

    for (const auto& p : funcOp.getArguments() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto val = p.value();

        log.nest().trace("Process argument #{0}", ind);

        const auto origType = val.getType().cast<vpux::NDTypeInterface>();
        const auto newType = newArgTypes[ind];

        if (newType == origType) {
            log.nest(2).trace("Nothing to change");
            continue;
        }

        log.nest(2).trace("Convert the argument type : '{0}' -> '{1}'", origType, newType);

        val.setType(newType);

        auto* firstUser = getFirstUser(val);
        if (firstUser == nullptr) {
            log.nest(2).trace("The argument has no users");
            continue;
        }

        OpBuilderLogger builderLog(log.nest(2));
        mlir::OpBuilder argBuilder(firstUser, &builderLog);

        auto* cvtOp = cvtOpBuilder(argBuilder, firstUser->getLoc(), val, origType);

        val.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    }

    //
    // Convert results
    //

    log.trace("Convert results");

    funcOp.walk([&](mlir::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        for (const auto& p : retOp->getOperands() | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());
            auto val = p.value();

            log.nest(2).trace("Process result #{0}", ind);

            const auto origType = val.getType();
            const auto newType = newResultTypes[ind].cast<vpux::NDTypeInterface>();

            if (newType == origType) {
                log.nest(3).trace("Nothing to change");
                continue;
            }

            log.nest(3).trace("Convert the result type : '{0}' -> '{1}'", newType, origType);

            auto* cvtOp = cvtOpBuilder(resBuilder, retOp.getLoc(), val, newType);

            retOp.setOperand(ind, cvtOp->getResult(0));
        }
    });

    return mlir::success();
}

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig vpux::getDefaultGreedyRewriteConfig() {
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 10;
    return config;
}

//
// appendLoc
//

mlir::Location vpux::appendLoc(mlir::Location baseLoc, StringRef suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, const formatv_object_base& suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, mlir::StringAttr suffix) {
    const mlir::Location suffixLoc = mlir::NameLoc::get(suffix);
    return mlir::FusedLoc::get(baseLoc.getContext(), {baseLoc, suffixLoc});
}

//
// BufferizeTypeConverter
//

vpux::BufferizeTypeConverter::BufferizeTypeConverter() {
    addConversion([](mlir::Type type) {
        return type;
    });

    addConversion(tensorToBuffer);
    addConversion(distributedTensorToBuffer);

    addConversion([&](VPU::SparseTensorType type) {
        const auto data = bufferizeTensor(type.getData());
        const auto sparsityMap = bufferizeTensor(type.getSparsityMap());
        const auto storageElementTable = bufferizeTensor(type.getStorageElementTable());

        VPUIP::CompressionSchemeAttr compressionScheme = nullptr;
        if (type.getCompressionScheme() != nullptr) {
            auto origCompression = type.getCompressionScheme();
            compressionScheme =
                    VPUIP::CompressionSchemeAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                      origCompression.getNumElems(), origCompression.getAlignment());
        }

        return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, type.getIsWeights(),
                                            compressionScheme);
    });

    addTargetMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addArgumentMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addSourceMaterialization(dummyConverter<mlir::TensorType>);

    addTargetMaterialization(dummyConverter<VPUIP::DistributedBufferType>);
    addArgumentMaterialization(dummyConverter<VPUIP::DistributedBufferType>);
    addSourceMaterialization(dummyConverter<VPU::DistributedTensorType>);

    addTargetMaterialization(dummyConverter<VPUIP::SparseBufferType>);
    addArgumentMaterialization(dummyConverter<VPUIP::SparseBufferType>);
    addSourceMaterialization(dummyConverter<VPU::SparseTensorType>);
}

//
// populateBufferizeMaterializationLegality
//

void vpux::populateBufferizeMaterializationLegality(mlir::ConversionTarget& target) {
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
}

//
// inferReturnTypes
//

void vpux::inferReturnTypes(mlir::Operation* op, InferShapedTypeMode mode) {
    auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(op);
    VPUX_THROW_WHEN(iface == nullptr, "Operation '{0}' doesn't implement InferTypeOpInterface", op->getName());

    SmallVector<mlir::Type> newTypes;
    VPUX_THROW_WHEN(iface.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                           op->getRegions(), newTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());

    for (auto p : zip(op->getResults(), newTypes)) {
        auto val = std::get<0>(p);
        auto newType = std::get<1>(p).dyn_cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(newType != nullptr, "newType has non vpux::NDTypeInterface type '{0}'", std::get<1>(p));

        if (!bitEnumContains(mode, InferShapedTypeMode::SHAPE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeShape(oldType.getShape());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::ELEM_TYPE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeElemType(oldType.getElementType());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::LAYOUT)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeDimsOrder(oldType.getDimsOrder());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::MEM_SPACE)) {
            if (const auto oldType = val.getType().dyn_cast<vpux::NDTypeInterface>()) {
                newType = newType.changeMemSpace(oldType.getMemSpace());
            }
        }

        val.setType(newType);
    }
}
