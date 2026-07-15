//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"

using namespace vpux;
using namespace IE;

void vpux::IE::arch37xx::registerElemTypeInfoOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::AffineReshapeOp::attachInterface<ElemTypeInfoAffineReshapeOpModel>(*ctx);
        IE::ClampOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ConcatOp::attachInterface<ElemTypeInfoConcatOpModel>(*ctx);
        IE::DepthToSpaceOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandDilatedOp::attachInterface<ElemTypeInfoExpandDilatedOpModel>(*ctx);
        IE::InterpolateOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::MaxPoolOp::attachInterface<ElemTypeInfoMaxPoolOpModel>(*ctx);
        IE::ReduceMaxOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ReorderOp::attachInterface<ElemTypeInfoReorderOpModel>(*ctx);
        IE::ReshapeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SliceOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SpaceToDepthOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SplitOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TileOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TransposeOp::attachInterface<ElemTypeInfoTransposeOpModel>(*ctx);
        IE::UnsqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::UpsamplingOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::PadOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::LayoutCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::PermuteCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ShapeCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
    });
}
