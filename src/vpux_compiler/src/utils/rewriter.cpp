//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/profiling/location.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <iterator>

using namespace vpux;

//
// updateFunctionSignature
//

mlir::LogicalResult vpux::updateFunctionSignature(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                                  ArrayRef<mlir::Type> newResultTypes, Logger log) {
    const auto origFuncType = funcOp.getFunctionType();

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

mlir::LogicalResult vpux::updateModuleInfo(mlir::Operation* module, ArrayRef<mlir::Type> newResultTypes, Logger log) {
    auto netInfoOps = module->getRegion(0).getOps<net::NetworkInfoOp>();
    if (netInfoOps.empty()) {
        log.trace("No NetworkInfoOp found in the module");
        return mlir::success();
    }
    if (std::next(netInfoOps.begin()) != netInfoOps.end()) {
        log.warning("Multiple NetworkInfoOp found in the module, only the first one will be updated");
    }
    auto netInfoOp = *netInfoOps.begin();
    // No actions on inputs, only outputs
    auto& outputsInfoBlock = netInfoOp.getOutputsInfo().front();

    // Collect all DataInfoOps to match with newResultTypes by index
    SmallVector<net::DataInfoOp> dataInfoOps;
    for (auto& op : outputsInfoBlock.getOperations()) {
        if (auto dataInfoOp = mlir::dyn_cast<net::DataInfoOp>(op)) {
            dataInfoOps.push_back(dataInfoOp);
        }
    }

    // Compare and update each DataInfoOp with corresponding newResultType
    for (size_t i = 0; i < dataInfoOps.size() && i < newResultTypes.size(); ++i) {
        auto dataInfoOp = dataInfoOps[i];
        auto newResultType = newResultTypes[i];

        auto primaryName = dataInfoOp.getName();
        auto currentUserType = dataInfoOp.getUserType();

        log.trace("Processing output DataInfoOp[{0}]: {1}, current type: {2}, new type: {3}", i, primaryName,
                  currentUserType, newResultType);

        // Compare the types - check if update is needed
        if (currentUserType != newResultType) {
            // Check if this is a scalar shape conversion case ([] -> [1])
            // In such cases, we should NOT update the DataInfo to preserve original metadata
            // This is because MLIR scalar is represented as [1], ngraph as [].
            // Keeping MLIR representation lead to fail in single-layer test on eltwise + scalar.
            bool isScalarShapeConversion = false;

            if (auto currentTensorType = mlir::dyn_cast<mlir::RankedTensorType>(currentUserType)) {
                if (auto newTensorType = mlir::dyn_cast<mlir::RankedTensorType>(newResultType)) {
                    // Check if this is scalar (rank 0) to [1] (rank 1) conversion
                    if (currentTensorType.getRank() == 0 && newTensorType.getRank() == 1) {
                        auto newShape = newTensorType.getShape();
                        if (newShape.size() == 1 && newShape[0] == 1) {
                            // This is a scalar [] -> [1] conversion, skip DataInfo update
                            isScalarShapeConversion = true;
                            log.trace("Skipping DataInfoOp[{0}] update - scalar shape conversion detected: [] "
                                      "-> [1]",
                                      i);
                        }
                    }
                }
            }

            if (!isScalarShapeConversion) {
                log.trace("Updating DataInfoOp[{0}] type: {1} -> {2}", i, currentUserType, newResultType);

                // Update the DataInfoOp with new type
                mlir::OpBuilder builder(dataInfoOp);
                builder.create<net::DataInfoOp>(dataInfoOp.getLoc(), dataInfoOp.getNameAttr(),
                                                mlir::TypeAttr::get(newResultType),  // Update with new type
                                                dataInfoOp.getOriginalShapeAttr(), dataInfoOp.getFriendlyNameAttr(),
                                                dataInfoOp.getInputNameAttr(), dataInfoOp.getTensorNamesAttr(),
                                                0  // Default profiling sections count
                );

                // Replace old DataInfoOp with new one
                dataInfoOp.erase();

                log.trace("Successfully updated DataInfoOp[{0}]", i);
            }
        } else {
            log.trace("DataInfoOp[{0}] type unchanged", i);
        }
    }

    // Check for size mismatch
    if (dataInfoOps.size() != newResultTypes.size()) {
        log.warning("Size mismatch: DataInfoOps count ({0}) != newResultTypes count ({1})", dataInfoOps.size(),
                    newResultTypes.size());
    }

    return mlir::success();
}

namespace {

// tensors -> buffers

mlir::bufferization::BufferLikeType tensorWithBoundsToBoundedBuffer(mlir::RankedTensorType tensorType) {
    VPUX_THROW_UNLESS(mlir::isa<Core::DynamicDimsMaskTensorType>(tensorType),
                      "Expected to have dynamic tensor, got {0}", tensorType);

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(tensorType);

    const auto dataMemShape = ndType.getShape();
    const auto dataMemOrder = ndType.getDimsOrder();
    const auto dataMemType = ndType.getElementType();
    const auto dataMemSpace = ndType.getMemSpace();
    const auto dataMemRef = getMemRefType(dataMemShape, dataMemType, dataMemOrder, dataMemSpace);

    const auto rank = checked_cast<int32_t>(dataMemShape.size());
    const auto si32 = getSInt32Type(tensorType.getContext());
    const auto dynamicShapeMemRef = getMemRefType({rank}, si32, DimsOrder::C, dataMemSpace);

    return VPUIP::BoundedBufferType::get(dataMemRef, dynamicShapeMemRef);
}

mlir::bufferization::BufferLikeType tensorToBuffer(mlir::RankedTensorType tensorType) {
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(tensorType)) {
        return tensorWithBoundsToBoundedBuffer(tensorType);
    }
    const auto type = mlir::cast<vpux::NDTypeInterface>(tensorType);
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return mlir::cast<mlir::bufferization::BufferLikeType>(getMemRefType(shape, elemType, order, memSpace));
}

mlir::bufferization::BufferLikeType distributedTensorToBuffer(VPU::DistributedTensorType type) {
    mlir::MLIRContext* ctx = type.getContext();
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(type)) {
        const auto data =
                VPUIP::DistributedBufferType::get(ctx, type.getShape().raw(), type.getElementType(), type.getOrder(),
                                                  type.getMemSpace(), type.getDistribution());
        const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

        const auto dataMemShape = ndType.getShape();
        const auto dataMemSpace = ndType.getMemSpace();
        const auto rank = checked_cast<int32_t>(dataMemShape.size());
        const auto si32 = getSInt32Type(ctx);

        const DimsOrder order = DimsOrder::C;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto memRefLayoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, ctx);

        const auto duplicatedMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        auto shapeDistribution = VPU::DistributionInfoAttr::get(ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr,
                                                                type.getDistribution().getNumClusters(), nullptr,
                                                                type.getDistribution().getUniformDistributedSegments(),
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto dynamicShapeMemRef =
                VPUIP::DistributedBufferType::get(ctx, {rank}, si32, memRefLayoutAttr, dataMemSpace, shapeDistribution);

        return VPUIP::BoundedBufferType::get(data, dynamicShapeMemRef);
    }

    return VPUIP::DistributedBufferType::get(ctx, type.getShape().raw(), type.getElementType(), type.getOrder(),
                                             type.getMemSpace(), type.getDistribution());
}

mlir::Type bufferizeTensor(mlir::Type tensorType) {
    if (tensorType == nullptr) {
        return nullptr;
    }

    if (auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType)) {
        return distributedTensorToBuffer(distributedType);
    } else if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
        return tensorToBuffer(rankedType);
    }
    VPUX_THROW("Unsupported type for bufferization '{0}'", tensorType);
    return nullptr;
}

VPUIP::SparseBufferType sparseTensorToBuffer(VPU::SparseTensorType type) {
    const auto data = bufferizeTensor(type.getData());
    const auto sparsityMap = bufferizeTensor(type.getSparsityMap());
    const auto storageElementTable = bufferizeTensor(type.getStorageElementTable());
    const auto seAttr = type.getSeAttr();

    VPUIP::SparsityCompressionAttr sparsityCompression = nullptr;
    if (auto origCompression = type.getSparsityCompression()) {
        sparsityCompression =
                VPUIP::SparsityCompressionAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                    origCompression.getNumElems(), origCompression.getAlignment());
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, type.getIsWeights(),
                                        sparsityCompression, seAttr);
}

// tensors <- buffers

mlir::RankedTensorType bufferToTensor(mlir::MemRefType memrefType) {
    const auto type = mlir::cast<NDTypeInterface>(memrefType);
    const auto encoding = getTensorAttr(type.getContext(), type.getDimsOrder(), type.getMemSpace());
    return mlir::RankedTensorType::get(type.getShape().raw(), type.getElementType(), encoding);
}

VPU::DistributedTensorType bufferToTensor(VPUIP::DistributedBufferType type) {
    // Note: distributed tensor is special: during bufferization its layout is
    // always an affine map (never vpux::MemRefAttr)
    const auto order = mlir::AffineMapAttr::get(type.getLayout().getAffineMap());
    return VPU::DistributedTensorType::get(type.getContext(), type.getShape().raw(), type.getElementType(), order,
                                           type.getMemSpace(), type.getDistribution());
}

mlir::Type tensorizeBuffer(mlir::Type bufferType) {
    if (bufferType == nullptr) {
        return nullptr;
    }

    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(bufferType)) {
        return bufferToTensor(distributedType);
    } else if (auto rankedType = mlir::dyn_cast<mlir::MemRefType>(bufferType)) {
        return bufferToTensor(rankedType);
    }
    VPUX_THROW("Unsupported buffer type for tensor conversion '{0}'", bufferType);
    return nullptr;
}

VPU::SparseTensorType bufferToTensor(VPUIP::SparseBufferType type) {
    const auto data = tensorizeBuffer(type.getData());
    const auto sparsityMap = tensorizeBuffer(type.getSparsityMap());
    const auto storageElementTable = tensorizeBuffer(type.getStorageElementTable());
    const auto seAttr = type.getSeAttr();

    VPU::SparsityCompressionAttr sparsityCompression = nullptr;
    if (auto origCompression = type.getSparsityCompression()) {
        sparsityCompression =
                VPU::SparsityCompressionAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                  origCompression.getNumElems(), origCompression.getAlignment());
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, type.getIsWeights(), sparsityCompression,
                                      seAttr);
}

mlir::RankedTensorType bufferToTensor(VPUIP::BoundedBufferType type) {
    auto dataType = tensorizeBuffer(type.getData());
    auto ndType = mlir::cast<NDTypeInterface>(dataType);
    auto dimsMask = DynamicDimsMask(ndType.getShape().size(), 1);
    dataType = Core::DynamicDimsMaskTensorType::get(ndType, dimsMask);
    return mlir::cast<mlir::RankedTensorType>(dataType);
}

mlir::Value materializeToTensor(mlir::OpBuilder& builder, mlir::TensorType type, mlir::ValueRange inputs,
                                mlir::Location loc) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "Expected only one input");
    VPUX_THROW_UNLESS(mlir::isa<mlir::BaseMemRefType>(inputs[0].getType()), "Expected input type is BaseMemRefType");
    return builder.create<mlir::bufferization::ToTensorOp>(loc, type, inputs[0]);
};

mlir::Value materializeToMemref(mlir::OpBuilder& builder, mlir::BaseMemRefType type, mlir::ValueRange inputs,
                                mlir::Location loc) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "Expected only one input");
    VPUX_THROW_UNLESS(mlir::isa<mlir::TensorType>(inputs[0].getType()), "Expected input type is TensorType");
    return builder.create<mlir::bufferization::ToBufferOp>(loc, type, inputs[0]);
};

}  // namespace

//
// convertFunc
//

mlir::LogicalResult vpux::convertFunc(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                      ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.getSymName());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.getSymName());
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

        const auto origType = mlir::cast<vpux::NDTypeInterface>(val.getType());
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

        auto* cvtOp = cvtOpBuilder(argBuilder, getValueLocation(val), val, origType);

        val.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    }

    //
    // Convert results
    //

    log.trace("Convert results");
    auto moduleOp = getModuleOp(funcOp);
    auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    SmallVector<net::DataInfoOp> outputsInfo;
    if (netInfoOps.size() == 1) {
        outputsInfo = to_small_vector(netInfoOps.front().getOutputsInfo().getOps<net::DataInfoOp>());
    } else {
        log.warning("Can't get location for output. If it isn't a test, please, debug this.");
    }

    funcOp.walk([&](mlir::func::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        for (const auto& p : retOp->getOperands() | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());
            auto val = p.value();

            log.nest(2).trace("Process result #{0}", ind);

            const auto origType = val.getType();
            const auto newType = mlir::cast<vpux::NDTypeInterface>(newResultTypes[ind]);

            if (newType == origType) {
                log.nest(3).trace("Nothing to change");
                continue;
            }

            log.nest(3).trace("Convert the result type : '{0}' -> '{1}'", newType, origType);

            mlir::Location cvtLoc = mlir::UnknownLoc::get(retOp.getContext());
            if (outputsInfo.empty()) {
                cvtLoc = appendLoc(getValueLocation(val), "out_{0}", p.index());
            } else {
                cvtLoc = outputsInfo[p.index()]->getLoc();
            }
            auto* cvtOp = cvtOpBuilder(resBuilder, cvtLoc, val, newType);

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
    config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Normal;
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
    VPUX_THROW_WHEN(suffix.getValue().find(LOCATION_ORIGIN_SEPARATOR) != std::string::npos,
                    "'{0}' character is reserved inside locations", LOCATION_ORIGIN_SEPARATOR);
    const mlir::Location suffixLoc = mlir::NameLoc::get(suffix);
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(baseLoc)) {
        const auto metadata = fusedLoc.getMetadata();
        auto locations = fusedLoc.getLocations().vec();
        locations.push_back(suffixLoc);
        return mlir::FusedLoc::get(baseLoc.getContext(), locations, metadata);
    }
    return mlir::FusedLoc::get(baseLoc.getContext(), {baseLoc, suffixLoc});
}

//
// BufferizeTypeConverterBase
//

vpux::BufferizeTypeConverterBase::BufferizeTypeConverterBase() {
    addConversion([](mlir::Type type) {
        return type;
    });

    addConversion(tensorToBuffer);
    addConversion(distributedTensorToBuffer);
    addConversion(sparseTensorToBuffer);
}

//
// BufferizeTypeConverter
//

vpux::BufferizeTypeConverter::BufferizeTypeConverter() {
    addTargetMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addArgumentMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addSourceMaterialization(dummyConverter<mlir::TensorType>);
}

//
// BufferizeOneShotTypeConverter
//

vpux::BufferizeOneShotTypeConverter::BufferizeOneShotTypeConverter() {
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization(materializeToMemref);
}

mlir::bufferization::OneShotBufferizationOptions vpux::getOneShotBufferizationOptions() {
    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowUnknownOps = true;
    options.copyBeforeWrite = false;
    // E#118032: Setting testAnalysisOnly as true does not introduce `bufferization.alloc_tensor`
    // but only adding `__inplace_operands_attr__`. Need to investigate whether it could be set to false,
    // and eliminate the introduced `bufferization.alloc_tensor, which will reduce the insertion of VPUIP.Copy
    options.testAnalysisOnly = true;
    options.unknownTypeConverterFn = [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
                                        const mlir::bufferization::BufferizationOptions& /*options*/) {
        return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);
    };
    options.defaultMemorySpaceFn = [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
        if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
            return getMemorySpace(rankedType);
        }
        return nullptr;
    };
    options.opFilter
            .allowDialect<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect,
                          mlir::func::FuncDialect, VPU::VPUDialect, Const::ConstDialect, mlir::linalg::LinalgDialect,
                          Core::CoreDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

    return options;
}

//
// getBufferType
//

vpux::NDTypeInterface vpux::getBufferType(mlir::Type tensorType) {
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return mlir::cast<vpux::NDTypeInterface>(tensorType);
    }

    return llvm::TypeSwitch<mlir::Type, mlir::Type>(tensorType)
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType rankedTensorType) -> mlir::Type {
                const auto encoding = vpux::getTensorAttr(rankedTensorType);
                if (encoding != nullptr) {
                    return tensorToBuffer(rankedTensorType);
                }
                return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(rankedTensorType, nullptr);
            })
            .Case<VPU::DistributedTensorType>([&](VPU::DistributedTensorType distributedTensorType) {
                return distributedTensorToBuffer(distributedTensorType);
            })
            .Case<VPU::SparseTensorType>([&](VPU::SparseTensorType sparseTensorType) {
                return sparseTensorToBuffer(sparseTensorType);
            })
            .Default([&](mlir::Type type) -> mlir::Type {
                // this is likely an unranked tensor type and we don't know how
                // to get a memSpace for it.
                const mlir::Attribute unknownMemSpace = nullptr;
                // E#108407: use UnknownTypeConverterFn directly, once it
                // accepts mlir::Type
                return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(mlir::cast<mlir::TensorType>(type),
                                                                                  unknownMemSpace);
            });
}

vpux::NDTypeInterface vpux::getBufferType(mlir::Value value) {
    return vpux::getBufferType(value.getType());
}

mlir::Type vpux::reconstructTensorType(mlir::Type type) {
    VPUX_THROW_UNLESS(isBufferType(type), "Provided a non-buffer type: {0}", type);

    // Note: enumerates types that are of interest to NPU compiler
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::MemRefType>([](mlir::MemRefType memref) {
                return bufferToTensor(memref);
            })
            .Case<mlir::UnrankedMemRefType>([](mlir::UnrankedMemRefType memref) {
                return mlir::UnrankedTensorType::get(memref.getElementType());
            })
            .Case<VPUIP::DistributedBufferType>([](VPUIP::DistributedBufferType memref) {
                return bufferToTensor(memref);
            })
            .Case<VPUIP::SparseBufferType>([](VPUIP::SparseBufferType memref) {
                return bufferToTensor(memref);
            })
            .Case<VPUIP::BoundedBufferType>([](VPUIP::BoundedBufferType memref) {
                return bufferToTensor(memref);
            })
            .Default([](mlir::Type memref) {
                VPUX_THROW("Unknown memref type: {0}", memref);
                return mlir::NoneType::get(memref.getContext());
            });
}

//
// getBuffer
//

mlir::Value vpux::getBuffer(mlir::RewriterBase& rewriter, mlir::Value value) {
    if (auto toTensorOp = value.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        return toTensorOp.getBuffer();
    }

    const auto tensorType = value.getType();
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return value;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(value);

    auto bufferType = vpux::getBufferType(value);
    auto origType = mlir::cast<vpux::NDTypeInterface>(value.getType());
    VPUX_THROW_WHEN(origType.hasRank() && origType.getRank() != bufferType.getRank(),
                    "Incompatible ranks: original rank {0}, buffer rank {1}", origType.getRank(), bufferType.getRank());

    // E#109609: replace with getResult()/getMemref() once we can convert
    //           VPUIP::{Distributed, Sparse}BufferType to mlir::BaseMemRefType
    return rewriter.create<mlir::bufferization::ToBufferOp>(value.getLoc(), bufferType, value)->getResult(0);
}

//
// bufferizeOperands
//

SmallVector<mlir::Value> vpux::bufferizeOperands(mlir::RewriterBase& rewriter, mlir::OperandRange operands) {
    if (operands.size() == 0) {
        return {};
    }
    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(llvm::size(operands));
    for (const auto& operand : operands) {
        auto buffer = vpux::getBuffer(rewriter, operand);
        newOperands.push_back(buffer);
    }
    return newOperands;
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
                                           op->getPropertiesStorage(), op->getRegions(), newTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());

    for (auto p : zip(op->getResults(), newTypes)) {
        auto val = std::get<0>(p);
        auto newType = mlir::dyn_cast<vpux::NDTypeInterface>(std::get<1>(p));
        VPUX_THROW_UNLESS(newType != nullptr, "newType has non vpux::NDTypeInterface type '{0}'", std::get<1>(p));

        if (!bitEnumContains(mode, InferShapedTypeMode::SHAPE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeShape(oldType.getShape());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::ELEM_TYPE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeElemType(oldType.getElementType());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::LAYOUT)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeDimsOrder(oldType.getDimsOrder());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::MEM_SPACE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeMemSpace(oldType.getMemSpace());
            }
        }

        val.setType(newType);
    }
}
