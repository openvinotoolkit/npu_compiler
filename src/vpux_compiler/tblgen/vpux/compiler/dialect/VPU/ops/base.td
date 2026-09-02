//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_BASE
#define VPUX_COMPILER_DIALECT_VPU_OPS_BASE

include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPU_Dialect,
            mnemonic,
            traits
        >;

class VPU_LayerOp<string mnemonic, list<Trait> traits = []> :
        VPU_Op<
            mnemonic,
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;

    code baseExtraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{)
            );
        }
    }];
    let extraClassDeclaration = baseExtraClassDeclaration;

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

// Base class for SW layers that access their inputs/outputs exclusively through DMA.
// Such ops keep their IOs in DDR and use a single CMX-resident aux_buffer as the
// SHAVE kernel workspace, duplicated across clusters. The shared bodies for
// SWOpInterface, AuxiliaryBufferOpInterface and ClusteredOpInterface are emitted once
// via extraClassDefinition, and rely on every derived op exposing exactly one
// `aux_buffer` operand (getAuxBuffer/getAuxBufferMutable).
//
// checkStrategyCompatibilityBody is overridable for ops requiring a different behavior
class VPU_SwIoDmaOp<string mnemonic, list<Trait> extraTraits = []> :
        VPU_LayerOp<
            mnemonic,
            [
                VPU_SwIoDmaOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["getDistributedTypeForOpOperand",
                                                                     "getDistributedTypeForOpResult"]>
            ] # extraTraits
        > {
    // Body of checkStrategyCompatibility(strategy, numTiles). Clustering is
    // supported by default; ops that must run on a single tile override this.
    code checkStrategyCompatibilityBody = [{
            return strategy == vpux::VPU::MultiClusterStrategy::Clustering;
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let extraClassDefinition = [{
        bool $cppClass::fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/, Byte /*reservedMem*/) {
            // IOs stay in DDR: the SHAVE kernel manages its own DMA transfers to CMX at runtime.
            return false;
        }

        bool $cppClass::fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
            return fitIntoCMX(buffers, vpux::Byte(0));
        }

        bool $cppClass::supportCycleCostCalculation() {
            return false;
        }

        vpux::SmallVector<mlir::OpOperand*> $cppClass::getAuxiliaryBuffers() {
            return {&getAuxBufferMutable()};
        }

        bool $cppClass::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy, size_t /*numTiles*/) {
            }] # checkStrategyCompatibilityBody # [{
        }

        vpux::VPU::DistributionInfo $cppClass::getExplicitDistributionInfoAttr(
                vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ::llvm::ArrayRef<int64_t> numTiles,
                const int64_t numClusters, ::llvm::ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
                const vpux::VPU::OverlapDistributionParams& overlapParams,
                const std::optional<::llvm::ArrayRef<int64_t>> /*memoryNumTiles*/) {
            return vpux::VPU::getSWExplicitDistributionInfo(mlir::cast<vpux::VPU::SWOpInterface>(getOperation()), shape,
                                                            distributionMode, numTiles, numClusters, alignment,
                                                            uniformDistributedSegments, overlapParams);
        }

        vpux::NDTypeInterface $cppClass::getDistributedTypeForOpOperand(
                mlir::OpOperand& operand, bool hasExplicitDistributedAttr,
                vpux::VPU::SiblingOpsAnalysis& siblingsAnalysis) {
            auto clusteredOp = mlir::cast<vpux::VPU::ClusteredOpInterface>(getOperation());
            // aux_buffer is DUPLICATED across all clusters as CMX workspace; IOs stay in DDR.
            if (operand.get() == getAuxBuffer()) {
                return vpux::VPU::getDistributedTypeFromInput(
                        clusteredOp, operand.get(), vpux::VPU::DistributionMode::DUPLICATED, {}, {},
                        vpux::VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr, siblingsAnalysis);
            }
            return mlir::cast<vpux::NDTypeInterface>(operand.get().getType());
        }

        vpux::NDTypeInterface $cppClass::getDistributedTypeForOpResult(
                mlir::Value result, vpux::VPU::MultiClusterStrategy /*strategy*/,
                vpux::VPU::SiblingOpsAnalysis& /*siblingsAnalysis*/, bool /*hasExplicitDistributedAttr*/) {
            // output stays in DDR:  kernel manages DMA from DDR to CMX internally.
            return mlir::cast<vpux::NDTypeInterface>(result.getType());
        }
    }];
}

#endif
