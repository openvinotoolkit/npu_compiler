//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape);
Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment);

bool needsPadding(const int64_t dim);

mlir::Value expandWithOffset(mlir::Location loc, mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                             IE::SliceOp sliceOp, mlir::Value expandValue, ShapeRef inPadsEnd, size_t expandDim);
mlir::Value paddingChannel(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                           ShapeRef filterPadsEnd, size_t expandDim, StringRef suffix);
mlir::Value paddingFilter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                          Shape filterPadsEnd);
SmallVector<int64_t> extractMeaningfulOutput(mlir::Operation* origOp, ShapeRef outPadsEnd);

mlir::Value generateZeroConst(mlir::Location loc, vpux::NDTypeInterface inType, ShapeRef padShape,
                              mlir::PatternRewriter& rewriter);

mlir::Value concatWithZeroConst(mlir::Location loc, mlir::Value filter, ShapeRef subInput, int64_t sliceChannelOffset,
                                mlir::PatternRewriter& rewriter);

mlir::Value padConvFilter(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, const int64_t inChanPadEnd,
                          const int64_t outChanPadEnd, const Logger& log);

bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* op);

// convert expand op to convolution utils
enum class ReshapeMode { RESHAPE_H_TO_C, PAD_W_RESHAPE, PAD_H_RESHAPE };

int64_t calculateAlignmentRequirementForExpandOpConversion(const vpux::NDTypeInterface expandInType);
int64_t getRequiredExpandConvInputAlignment(IE::ExpandOp expandOp, vpux::NDTypeInterface expandInType,
                                            ShapeRef expandInShape, vpux::NDTypeInterface expandOutType,
                                            ShapeRef expandOutShape, mlir::Type convOutElemType,
                                            IE::ReshapeMode reshapeMode);
bool beneficialToPadHeight(IE::ExpandOp origOp);
bool beneficialToPadHeight(vpux::NDTypeInterface expandInType);
bool beneficialToPadWidth(IE::ExpandOp origOp);
bool beneficialToPadWidth(vpux::NDTypeInterface expandInType, mlir::Value origExpandOutput);
bool beneficialToReshapeHeightToChannel(IE::ExpandOp origOp);
bool beneficialToReshapeHeightToChannel(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType);
void convertExpandTypesToSupportedLayout(IE::ExpandOp expandOp, const vpux::DimsOrder& supportedLayout,
                                         vpux::NDTypeInterface& expandInType, vpux::NDTypeInterface& expandOutType,
                                         SmallVector<int64_t>& padsBegin, SmallVector<int64_t>& padsEnd);
bool isEligibleConvertToConv(IE::ExpandOp expandOp, Logger log, StringRef debugName);
bool isEligibleConvertToConv(mlir::Value origExpandOutput, vpux::NDTypeInterface expandInType,
                             vpux::NDTypeInterface expandOutType, ArrayRef<int64_t> expandPadsBegin,
                             ArrayRef<int64_t> expandPadsEnd, mlir::ModuleOp moduleOp, mlir::Location loc, Logger log,
                             StringRef debugName);
std::optional<vpux::Dim> getExpandAxis(IE::ExpandOp expandOp);
}  // namespace IE
}  // namespace vpux
