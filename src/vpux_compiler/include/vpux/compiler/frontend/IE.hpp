//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
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

namespace vpux {
namespace IE {

struct ImportNetworkConfig {
    bool sharedConstants = false;
    bool enableProfiling = false;
    DummyOpMode stubLayers = DummyOpMode::DISABLED;
    bool dynamicShapeToStatic = false;
    bool enableWeightsSeparationPath = false;
    bool enableDecomposeSDPA = false;
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
                                     DummyOpMode stubLayers, bool dynamicShapeToStatic);
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
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    static Callback getParser(const std::shared_ptr<ov::Node>& op);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

    void parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Constant>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ConvertLike>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LogSoftmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tile>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Relu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Split>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Power>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Multiply>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GroupConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset12::GroupNormalization>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GroupConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset16::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShuffleChannels>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GatherTree>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::OneHot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::BatchNormInference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::GatherElements>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ScatterNDUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ScatterUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset12::ScatterElementsUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ScatterElementsUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMean>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceLogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceLogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceProd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL1>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Mod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reverse>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::FakeConvert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Cos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sqrt>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Cosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Asinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Acosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Atanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Log>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Selu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Floor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Round>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Mish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Erf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Broadcast>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Bucketize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::NormalizeL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::CumSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode);
    void parseNode(mlir::OpBuilder& builder,
                   const std::shared_ptr<ov::opset6::ExperimentalDetectronROIFeatureExtractor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::StridedSlice>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PRelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Swish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Negative>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::CTCGreedyDecoder>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::CTCGreedyDecoderSeqLen>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Pad>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LSTMCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::LSTMCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Subtract>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LSTMSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Ceiling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Equal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Select>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode);
    void parseNode(mlir::OpBuilder& builder,
                   const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DepthToSpace>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReverseSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Less>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LessEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::NotEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::SoftPlus>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Greater>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GreaterEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseXor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseNot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalNot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalXor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::SpaceToDepth>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::BatchToSpace>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::SpaceToBatch>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ExtractImagePatches>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Abs>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Atan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Asin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Acos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Roll>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::HSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::HardSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::EmbeddingBagOffsetsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::EmbeddingSegmentsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::EmbeddingBagPackedSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::GRUCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::GRUSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DeformablePSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TensorIterator>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Loop>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::IDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset15::STFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShapeOf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Range>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::NonZero>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RMS>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RoPE>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::Inverse>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::DeformableConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::VariadicSplit>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset13::ScaledDotProductAttention>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v16::Identity>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::Op>& origNode);

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
void NGraphImporter::parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    auto targetPtr = std::dynamic_pointer_cast<NodeType>(origNode);
    OPENVINO_ASSERT(targetPtr != nullptr);
    parseNode(builder, targetPtr);
}

}  // namespace IE
}  // namespace vpux
