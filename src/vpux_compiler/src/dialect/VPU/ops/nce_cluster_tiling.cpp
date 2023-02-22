//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchOpInterface
//

mlir::OperandRange vpux::VPU::NCEClusterTilingOp::getSuccessorEntryOperands(unsigned index) {
    VPUX_THROW_UNLESS(index == 0, "Invalid region index: {0}", index);
    return operands();
}

void vpux::VPU::NCEClusterTilingOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<mlir::Attribute>,
                                                        SmallVectorImpl<mlir::RegionSuccessor>& regions) {
    if (index.hasValue()) {
        VPUX_THROW_UNLESS(*index == 0, "Invalid region index: {0}", *index);
        regions.push_back(mlir::RegionSuccessor(results()));
        return;
    }

    regions.push_back(mlir::RegionSuccessor(&body(), body().getArguments()));
}

//
// Inner info
//

mlir::Operation* vpux::VPU::NCEClusterTilingOp::getInnerTaskOp() {
    return &body().front().getOperations().front();
}

//
// print/parse
//

void vpux::VPU::print(mlir::OpAsmPrinter& p, VPU::NCEClusterTilingOp op) {
    // (%operand as %blockArg: <type>, ...)

    auto& body = op.body();
    VPUX_THROW_UNLESS(!body.empty(), "Cannot serialize operation with empty body.");

    auto* entry = &op.body().front();
    VPUX_THROW_UNLESS(op.getNumOperands() == entry->getNumArguments(),
                      "Mismatch between the number of operands({0}) and body arguments({1}).", op.getNumOperands(),
                      entry->getNumArguments());

    p << " (";
    llvm::interleaveComma(op.operands(), p, [&, n = 0](mlir::Value operand) mutable {
        auto argument = entry->getArgument(n++);
        p << operand << " as " << argument << ": " << argument.getType();
    });
    p << ")";

    p.printOptionalAttrDictWithKeyword(op->getAttrs());
    p.printOptionalArrowTypeList(op.getResultTypes());
    p << " ";
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

mlir::ParseResult vpux::VPU::parseNCEClusterTilingOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    // Parse operands (%operand as %blockArg : <type>).
    SmallVector<mlir::OpAsmParser::OperandType> operands;
    SmallVector<mlir::OpAsmParser::OperandType> blockArgs;
    SmallVector<mlir::Type> operandRawTypes;
    SmallVector<mlir::Type> blockTypes;

    // Parse a single instance of `%operand as %blockArg : <type>`.
    auto parseOperands = [&]() -> mlir::ParseResult {
        if (parser.parseOperand(operands.emplace_back()) || parser.parseKeyword("as") ||
            parser.parseOperand(blockArgs.emplace_back()) || parser.parseColonType(blockTypes.emplace_back())) {
            return mlir::failure();
        }

        operandRawTypes.push_back(mlir::Type{});
        return mlir::success();
    };

    auto argsLoc = parser.getCurrentLocation();
    if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::OptionalParen, parseOperands) ||
        parser.resolveOperands(operands, operandRawTypes, argsLoc, result.operands)) {
        return mlir::failure();
    }

    // Parse operation attributes.
    mlir::NamedAttrList attrs;
    if (parser.parseOptionalAttrDictWithKeyword(attrs)) {
        return mlir::failure();
    }
    result.addAttributes(attrs);

    // Parse operation results.
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs, blockTypes, /*argLocations=*/{}, /*enableNameShadowing=*/false)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// build
//

void vpux::VPU::NCEClusterTilingOp::build(mlir::OpBuilder& builder, mlir::OperationState& result,
                                          mlir::TypeRange resultTypes, mlir::ValueRange operands,
                                          BodyBuilderFn bodyBuilder) {
    result.addOperands(operands);
    result.addTypes(resultTypes);

    // Add a body region with block arguments
    auto* bodyRegion = result.addRegion();
    auto& bodyBlock = bodyRegion->emplaceBlock();
    for (auto operand : operands) {
        auto type = operand.getType();
        if (auto distributedType = type.dyn_cast<DistributedTensorType>()) {
            type = distributedType.getCompactType();
        } else if (auto sparseType = type.dyn_cast<SparseTensorType>()) {
            if (auto distDataType = sparseType.getData().dyn_cast<DistributedTensorType>()) {
                mlir::RankedTensorType dataType = distDataType.getCompactType();
                mlir::RankedTensorType smType = nullptr;
                if (sparseType.getSparsityMap() != nullptr &&
                    sparseType.getSparsityMap().isa<DistributedTensorType>()) {
                    smType = sparseType.getSparsityMap().cast<DistributedTensorType>().getCompactType();
                }
                mlir::RankedTensorType seType = nullptr;
                if (sparseType.getStorageElementTable() != nullptr &&
                    sparseType.getStorageElementTable().isa<DistributedTensorType>()) {
                    seType = sparseType.getStorageElementTable().cast<DistributedTensorType>().getCompactType();
                }
                type = SparseTensorType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                             sparseType.getCompressionScheme());
            }
        }

        bodyBlock.addArgument(type, operand.getLoc());
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);

    VPUX_THROW_UNLESS(bodyBuilder, "Got empty body builder.");
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(vpux::VPU::NCEClusterTilingOp op) {
    auto& body = op.body();
    if (!body.hasOneBlock()) {
        return errorAt(op->getLoc(), "Operation must have only one block.");
    }

    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        return errorAt(op->getLoc(),
                       "Operation must have at least one operand to satisfy pure no-side-effects semantic.");
    }

    auto bodyNumArgs = body.getNumArguments();
    if (numOperands != bodyNumArgs) {
        return errorAt(op->getLoc(), "Mismatch between the number of operands({0}) and body arguments({1}).",
                       numOperands, bodyNumArgs);
    }

    if (op->getNumResults() == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one result.");
    }

    const auto checkShape = [&](mlir::ValueRange operands) {
        for (auto operand : operands) {
            if (auto distributedTensor = operand.getType().dyn_cast<vpux::VPU::DistributedTensorType>()) {
                auto rank = distributedTensor.getShape().size();
                if (rank != 4) {
                    return errorAt(op->getLoc(), "Only 4D tensors are supported. Got {0}", rank);
                }
            }
        }

        return mlir::success();
    };

    auto isOperandsValid = checkShape(op->getOperands());
    if (isOperandsValid.failed()) {
        return mlir::failure();
    }

    auto isArgsValid = checkShape(op.body().getArguments());
    if (isArgsValid.failed()) {
        return mlir::failure();
    }

    auto yieldOps = op.body().getOps<vpux::VPU::YieldOp>();
    const auto numYieldOps = std::distance(yieldOps.begin(), yieldOps.end());
    if (numYieldOps != 1) {
        return errorAt(op->getLoc(), "Operation have to contain one YieldOp, but it has {0}", numYieldOps);
    }

    return mlir::success();
}

namespace {

/// @brief Finds and eliminates sequences of surplus Copies that effectively leave the Type unchanged
/// @details The pattern of surplus Copy chains can appear when two consequent operations are ClusterTiled the same way:
/// for example, when a (Conv)->(Conv) chain is all split-over-height
/// @example The expected pattern is:
/// %1 = VPU.NCEClusterTiling(%0 as %arg0 : !Type1) -> !Type2 {
///     %3 = IE.Copy(%arg0)
///     VPU.Yield %3
/// }
/// %2 = VPU.NCEClusterTiling(%1 as %arg0 : !Type2) -> !Type1 {
///     %3 = IE.Copy(%arg0)
///     VPU.Yield %3
/// }
/// Action expected: replace %2 usage with %0, eliminate the unused CopyOps
class EliminateCopyPairs final : public mlir::OpRewritePattern<VPU::NCEClusterTilingOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult EliminateCopyPairs::matchAndRewrite(VPU::NCEClusterTilingOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    // This particular ClusterTilingOp should contain a single Copy operation
    auto copyOps = origOp.body().getOps<vpux::VPU::CopyOp>();
    if (std::distance(copyOps.begin(), copyOps.end()) != 1) {
        return mlir::failure();
    }

    // Its input should be produced by a ClusterTilingOp that contains a single Copy operation
    auto producerClusterTilingOp = origOp.operands()[0].getDefiningOp<VPU::NCEClusterTilingOp>();
    if (producerClusterTilingOp == nullptr) {
        return mlir::failure();
    }
    auto producerCopyOps = producerClusterTilingOp.body().getOps<vpux::VPU::CopyOp>();
    if (std::distance(producerCopyOps.begin(), producerCopyOps.end()) != 1) {
        return mlir::failure();
    }

    // The I/O types of this CopyOp-chain should be similar
    auto producerInput = producerClusterTilingOp.getOperand(0);
    auto output = origOp.getResult(0);

    if (producerInput.getType() != output.getType()) {
        const auto inDistributedTypeInterface = producerInput.getType().dyn_cast<VPU::DistributedTypeInterface>();
        const auto outDistributedTypeInterface = output.getType().dyn_cast<VPU::DistributedTypeInterface>();

        if (inDistributedTypeInterface == nullptr || outDistributedTypeInterface == nullptr ||
            !inDistributedTypeInterface.containsDistributedTypes() ||
            !outDistributedTypeInterface.containsDistributedTypes()) {
            return mlir::failure();
        }

        if (VPU::isDistributedCastCompatible(
                    inDistributedTypeInterface.getDistributedTypes().front().cast<VPU::DistributedTensorType>(),
                    outDistributedTypeInterface.getDistributedTypes().front().cast<VPU::DistributedTensorType>())
                    .failed()) {
            return mlir::failure();
        }

        const auto distributedCastOp =
                rewriter.create<VPU::DistributedCastOp>(origOp.getLoc(), output.getType(), producerInput);

        rewriter.replaceOp(origOp, distributedCastOp->getResult(0));
        return mlir::success();
    }

    rewriter.replaceOp(origOp, producerInput);
    return mlir::success();
}

class RemoveIfEmptyBody final : public mlir::OpRewritePattern<VPU::NCEClusterTilingOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RemoveIfEmptyBody::matchAndRewrite(VPU::NCEClusterTilingOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    if (origOp.operands().size() != 1 || origOp.results().size() != 1) {
        return mlir::failure();
    }

    auto input = origOp.getOperand(0);
    auto output = origOp.getResult(0);

    if (input.getType() != output.getType()) {
        return mlir::failure();
    }

    auto& bodyBlock = origOp.body().front();

    // Check if body does not have any operation besides YieldOp which is
    // integral part of NCEClusterTiling
    for (auto& op : bodyBlock.getOperations()) {
        if (!mlir::isa<VPU::YieldOp>(op)) {
            return mlir::failure();
        }
    }

    rewriter.replaceOp(origOp, input);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::NCEClusterTilingOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* ctx) {
    patterns.add<EliminateCopyPairs>(ctx);
    patterns.add<RemoveIfEmptyBody>(ctx);
}
