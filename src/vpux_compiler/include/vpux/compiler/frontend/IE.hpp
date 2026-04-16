//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/utils/IE/hash.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

// Opset versions supported
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset14.hpp>
#include <openvino/opsets/opset15.hpp>
#include <openvino/opsets/opset16.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset9.hpp>

#include <ov_ops/nms_ie_internal.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include <intel_npu/ops/flash_attention_tile.hpp>

namespace vpux {
namespace IE {

struct ImportNetworkConfig {
    bool sharedConstants = false;
    bool enableProfiling = false;
    DummyOpMode stubLayers = DummyOpMode::DISABLED;
    bool dynamicShapeToStatic = false;
    bool enableWeightsSeparationPath = false;
    bool enableDecomposeSDPA = false;
    bool executionModeAccuracy = false;
    std::set<std::string> ioWithDynamicStrides;
};

// TODO Get rid of this function (importNetwork), move logic to compiler.cpp
mlir::OwningOpRef<mlir::ModuleOp> importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
                                                const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                                const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                                mlir::TimingScope& rootTiming, const ImportNetworkConfig& importCfg,
                                                Logger log = Logger::global());

std::vector<std::shared_ptr<const ov::Node>> buildOVParams(const std::shared_ptr<const ov::Model>& model);
std::vector<std::shared_ptr<const ov::Node>> buildOVResults(const std::shared_ptr<const ov::Model>& model);

// TODO Move to separate file NGraphPasses
class NGraphPasses final {
public:
    static void runNGraphPasses(const std::shared_ptr<ov::Model>& netGraph, mlir::TimingScope& rootTiming,
                                const ImportNetworkConfig& importCfg);
};

class NGraphImporter final {
public:
    using OrigNode = ov::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;

    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ov::Model> netGraph, bool sharedConstants, Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    mlir::func::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName, mlir::TimingScope& rootTiming,
                                     const ImportNetworkConfig& importCfg);
    void buildBlockFromRegion(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block);
    void buildBlockFromBody(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block);
    SmallVector<mlir::Type> getRegionResults();
    SmallVector<mlir::Type> getLoopLikeRegionResults(int64_t numIter, int32_t numResults,
                                                     ArrayRef<mlir::Attribute> concatOutputVector,
                                                     ArrayRef<mlir::Attribute> invariantOutputVector);
    static bool isOpSupported(const std::shared_ptr<ov::Node>& op);

    // Bounds related logic
    void saveBoundsInfoForInput(::mlir::BlockArgument& funcInputVal, const ov::PartialShape& partialShape);
    bool isUpperBoundsMissing(const OrigNodePtr& origNode);
    static bool hasValidBounds(const ov::PartialShape& partialShape);

private:
    using NodeOutputMap = std::unordered_map<ov::Output<OrigNode>, mlir::Value>;
    using Callback = mlir::Operation* (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void extractPrecisionInfo(mlir::OpBuilder& moduleBuilder, const OrigNodePtr& origNode,
                              const ImportNetworkConfig& importCfg, mlir::Operation* op,
                              std::optional<net::PrecisionRequirementOp>& precReqOp);
    static Callback getParser(const std::shared_ptr<ov::Node>& op);
    template <class NodeType>
    mlir::Operation* parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    mlir::Operation* parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
        return nullptr;
    }

    mlir::Operation* parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Constant>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convert>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ConvertLike>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LogSoftmax>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tile>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Relu>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Split>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Power>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Multiply>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convolution>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GroupConvolution>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset12::GroupNormalization>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::ConvolutionBackpropData>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::GroupConvolutionBackpropData>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::AvgPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset16::AvgPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MaxPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::MaxPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::MaxPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShuffleChannels>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GatherTree>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::OneHot>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset16::OneHot>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset5::BatchNormInference>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::GatherElements>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ScatterNDUpdate>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ScatterUpdate>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset12::ScatterElementsUpdate>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::ScatterElementsUpdate>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Clamp>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Elu>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reshape>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Squeeze>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sigmoid>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LRN>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMax>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMean>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceLogicalOr>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceLogicalAnd>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceProd>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceSum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMin>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL1>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL2>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Unsqueeze>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Minimum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Maximum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Add>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Divide>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::SquaredDifference>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::FloorMod>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Mod>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Proposal>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reverse>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::FakeQuantize>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::FakeConvert>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MatMul>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tan>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tanh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sin>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Cos>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sqrt>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sinh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Cosh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Asinh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Acosh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Atanh>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Log>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Selu>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Exp>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::HSwish>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Floor>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Round>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Mish>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Erf>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Broadcast>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Bucketize>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Transpose>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Interpolate>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::TopK>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TopK>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::RegionYolo>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ReorgYolo>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::NormalizeL2>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::CumSum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Concat>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ROIPooling>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PSROIPooling>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset6::ExperimentalDetectronROIFeatureExtractor>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::StridedSlice>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PRelu>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Swish>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GRN>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Negative>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sign>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::CTCGreedyDecoder>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset6::CTCGreedyDecoderSeqLen>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Pad>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LSTMCell>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::LSTMCell>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Subtract>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalAnd>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LSTMSequence>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Ceiling>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Equal>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Select>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DepthToSpace>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReverseSequence>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Less>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LessEqual>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::NotEqual>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::SoftPlus>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::SoftSign>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Greater>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GreaterEqual>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseAnd>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseOr>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseXor>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseNot>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalNot>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalOr>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalXor>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::SpaceToDepth>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::BatchToSpace>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::SpaceToBatch>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::ExtractImagePatches>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Abs>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Atan>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Asin>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Acos>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Roll>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::HSigmoid>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::HardSigmoid>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingBagOffsetsSum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingSegmentsSum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingBagPackedSum>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::GRUCell>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::GRUSequence>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::DeformablePSROIPooling>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TensorIterator>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Loop>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset16::ISTFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::IDFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset15::STFT>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShapeOf>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Range>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::NonZero>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RMS>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RoPE>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::Inverse>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset8::DeformableConvolution>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::VariadicSplit>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset13::ScaledDotProductAttention>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::intel_npu::op::FlashAttentionTile>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v16::Identity>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::Op>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset10::IsNaN>& origNode);
    mlir::Operation* parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset10::IsInf>& origNode);

    SmallVector<mlir::Value> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    void addOutputs(const OrigNodePtr& node, const std::vector<mlir::Value>& outputs);
    mlir::Location createLocation(const OrigNodePtr& node);

    mlir::RankedTensorType importTensor(const ov::PartialShape& shape, const ov::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ov::op::AutoBroadcastType bType);
    IE::BroadcastTypeAttr importBroadcastMode(ov::op::BroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ov::op::RoundingType roundingType);
    IE::EpsModeAttr importEpsMode(ov::op::EpsMode val);
    IE::MvnEpsModeAttr importMvnEpsMode(ov::op::MVNEpsMode val);
    IE::TopKModeAttr importTopKMode(ov::op::TopKMode val);
    IE::TopKSortTypeAttr importTopKSortType(ov::op::TopKSortType val);
    IE::GridSampleModeAttr importGridSampleMode(const ov::op::v9::GridSample::InterpolationMode& val);
    IE::GridSamplePaddingModeAttr importGridSamplePaddingMode(const ov::op::v9::GridSample::PaddingMode& val);
    IE::ProposalAttr importProposalAttrs(const ov::op::v0::Proposal::Attributes& val);
    IE::ReverseModeAttr importReverseMode(const ov::op::v1::Reverse::Mode mode);
    IE::OneHotModeAttr importOneHotMode(const ov::op::v16::OneHot::NegativeIndicesMode mode);
    IE::RoPEModeAttr importRoPEMode(const ov::op::internal::RoPE::Config& cfg);
    IE::InterpolateAttr importInterpolateAttrs(const ov::opset4::Interpolate::InterpolateAttrs& val);
    IE::DetectionOutputAttr importDetectionOutputAttrs(const ov::op::v0::DetectionOutput::Attributes& val);
    IE::ExperimentalDetectronROIFeatureExtractorAttr importExpDetectronROIFeatureExtractAttrs(
            const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& val);
    IE::ROIPoolingMethodAttr importROIPoolingMethod(const std::string& method);
    IE::PSROIPoolingModeAttr importPSROIPoolingMode(const std::string& mode);
    IE::ROIAlignMethodAttr importROIAlignMethod(const ov::op::v9::ROIAlign::PoolingMode& mode);
    IE::ROIAlignAlignedMethodAttr importROIAlignAlignedMethod(const ov::op::v9::ROIAlign::AlignedMode& mode);
    IE::PadModeAttr importPadMode(const ov::op::PadMode val);
    IE::RoundModeAttr importRoundMode(const ov::op::v5::Round::RoundMode val);
    IE::RNNSequenceDirectionAttr importRNNSequenceDirection(const ov::op::RecurrentSequenceDirection val);
    IE::BoxEncodingTypeAttr importBoxEncodingType(const int val);
    IE::DepthToSpaceModeAttr importDepthToSpaceMode(const ov::op::v0::DepthToSpace::DepthToSpaceMode val);
    IE::SpaceToDepthModeAttr importSpaceToDepthMode(const ov::op::v0::SpaceToDepth::SpaceToDepthMode val);
    IE::PadTypeAttr importPadType(ov::op::PadType autoPads);
    IE::DeformablePSROIPoolingModeAttr importDeformablePSROIPoolingMode(const std::string& mode);
    IE::ScatterElementsUpdateReductionTypeAttr importScatterElementsUpdateReductionType(
            ov::op::v12::ScatterElementsUpdate::Reduction val);
    IE::DetectionOutputCodeTypeAttr importDetectionOutputCodeType(const std::string& codeType);
    IE::SliceInputPortMapAttr importSliceInputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>& desc,
            const mlir::Value input, const int64_t numIter);
    IE::InvariantInputPortMapAttr importInvariantInputPortMapAttr(
            mlir::MLIRContext* ctx,
            const std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>& desc);
    IE::MergedInputPortMapAttr importMergedInputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>& desc);
    IE::ConcatOutputPortMapAttr importConcatOutputPortMapAttr(
            mlir::MLIRContext* ctx,
            const std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>& desc);
    IE::InvariantOutputPortMapAttr importBodyOutputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>& desc);
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ov::Model> _netGraph;
    bool _sharedConstants = false;
    Logger _log;

    NodeOutputMap _importedVals;
};

template <class NodeType>
mlir::Operation* NGraphImporter::parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    auto targetPtr = std::dynamic_pointer_cast<NodeType>(origNode);
    OPENVINO_ASSERT(targetPtr != nullptr);
    return parseNode(builder, targetPtr);
}

}  // namespace IE
}  // namespace vpux
