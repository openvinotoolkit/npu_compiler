//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/IR/Dialect.h>

namespace vpux {

// VPU-specific BufferizableOpInterface external model base
template <typename ConcreteModel, typename ConcreteOp>
class BufferizableOpInterfaceExternalModelBase :
        public mlir::bufferization::BufferizableOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    bool bufferizesToMemoryRead(mlir::Operation* op, mlir::OpOperand& opOperand,
                                const mlir::bufferization::AnalysisState& state) const {
        return static_cast<const ConcreteModel*>(this)->bufferizesToMemoryReadImpl(downcast(op), opOperand, state);
    }
    bool bufferizesToMemoryWrite(mlir::Operation* op, mlir::OpOperand& opOperand,
                                 const mlir::bufferization::AnalysisState& state) const {
        return static_cast<const ConcreteModel*>(this)->bufferizesToMemoryWriteImpl(downcast(op), opOperand, state);
    }
    mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation* op, mlir::OpOperand& opOperand,
                                                             const mlir::bufferization::AnalysisState& state) const {
        return static_cast<const ConcreteModel*>(this)->getAliasingValuesImpl(downcast(op), opOperand, state);
    }

    // Default BufferizableOpInterface::bufferize() implementation used to set
    // up bufferized operands and forward the arguments to specific model's
    // bufferizeImpl()
    mlir::LogicalResult bufferize(mlir::Operation* op, mlir::RewriterBase& rewriter,
                                  const mlir::bufferization::BufferizationOptions& options) const {
        auto bufferizedOperands = vpux::bufferizeOperands(rewriter, op->getOperands());
        auto adaptor = typename ConcreteOp::Adaptor(bufferizedOperands, op->getAttrDictionary(),
                                                    op->getPropertiesStorage(), op->getRegions());
        return static_cast<const ConcreteModel*>(this)->bufferizeImpl(downcast(op), rewriter, options, adaptor);
    }

protected:
    // Default implementations if not overridden by ConcreteModel
    bool bufferizesToMemoryReadImpl(ConcreteOp, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return true;
    }
    bool bufferizesToMemoryWriteImpl(ConcreteOp, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return true;
    }
    mlir::bufferization::AliasingValueList getAliasingValuesImpl(ConcreteOp, mlir::OpOperand&,
                                                                 const mlir::bufferization::AnalysisState&) const {
        return {};
    }

private:
    // Utility method for casting mlir::Operation* to ConcreteOp
    ConcreteOp downcast(mlir::Operation* op) const {
        VPUX_THROW_UNLESS(mlir::isa<ConcreteOp>(*op), "Operation {0} cannot be converted to ConcreteOp", op->getName());
        return mlir::cast<ConcreteOp>(*op);
    }
};

//
// BufferizableOpInterfaces registers
//

void registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry);
void registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry);
void registerFuncAndReturnBufferizableOpInterfaces(mlir::DialectRegistry& registry);
void registerVPUBufferizableOpInterfaces(mlir::DialectRegistry& registry);
void registerConstDeclareBufferizableOpInterfaces(mlir::DialectRegistry& registry);
void registerCoreBufferizableOpInterfaces(mlir::DialectRegistry& registry);

//
// bufferize vpu ops functions
//

mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::CopyOp origOp, VPU::CopyOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::ExpandOp origOp, VPU::ExpandOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::ConvertOp origOp, VPU::ConvertOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::StridedSliceOp origOp,
                                VPU::StridedSliceOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::SliceOp origOp, VPU::SliceOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::SplitOp origOp, VPU::SplitOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::PermuteCastOp origOp, VPU::PermuteCastOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::QuantizeCastOp origOp,
                                VPU::QuantizeCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::DistributedCastOp origOp,
                                VPU::DistributedCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::StubOp origOp, VPU::StubOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::GroupSparseTensorOp origOp,
                                VPU::GroupSparseTensorOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::UngroupSparseTensorOp origOp,
                                VPU::UngroupSparseTensorOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::StorageElementTableOp origOp,
                                VPU::StorageElementTableOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::ShapeCastOp origOp, VPU::ShapeCastOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::LayoutCastOp origOp, VPU::LayoutCastOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::UpsamplingOp origOp, VPU::UpsamplingOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::GatherDMAOp origOp, VPU::GatherDMAOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::ShapeOfOp origOp, VPU::ShapeOfOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);

template <typename ConcreteOp>
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, ConcreteOp origOp, typename ConcreteOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);

//
// bufferize nce ops functions
//

mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEConvolutionOp origOp,
                                VPU::NCEConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMaxPoolOp origOp, VPU::NCEMaxPoolOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEAveragePoolOp origOp,
                                VPU::NCEAveragePoolOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEDepthConvolutionOp origOp,
                                VPU::NCEDepthConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEInterpolateOp origOp,
                                VPU::NCEInterpolateOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEEltwiseOp origOp, VPU::NCEEltwiseOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEReduceOp origOp, VPU::NCEReduceOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCECompressConvolutionOp origOp,
                                VPU::NCECompressConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEPermuteOp origOp, VPU::NCEPermuteOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);
mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMatMulOp origOp, VPU::NCEMatMulOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);

//
// bufferize non-VPU operations
//

mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, Const::DeclareOp origOp, Const::DeclareOp::Adaptor& newArgs,
                                mlir::RewriterBase& rewriter);

mlir::LogicalResult bufferizeOp(mlir::MLIRContext* ctx, Core::ReinterpretCastOp origOp,
                                Core::ReinterpretCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter);

// generic VPU-specific one-shot bufferization model
template <typename ConcreteOp>
struct VpuGenericOneShotBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<VpuGenericOneShotBufferizeModel<ConcreteOp>, ConcreteOp> {
    mlir::LogicalResult bufferizeImpl(ConcreteOp op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions&,
                                      typename ConcreteOp::Adaptor& opAdaptor) const {
        // call the actual bufferization function
        return ::vpux::bufferizeOp(op.getContext(), op, opAdaptor, rewriter);
    }
};

}  // namespace vpux
