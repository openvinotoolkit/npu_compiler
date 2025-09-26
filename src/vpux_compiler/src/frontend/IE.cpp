//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/cal_range_data.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/net/network_info_utils.hpp"
#include "vpux/compiler/utils/range_bound.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif

#include "intel_npu/config/config.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/Verifier.h>

#include <openvino/core/node.hpp>
#include <openvino/core/rt_info/weightless_caching_attributes.hpp>
#include <openvino/core/type.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/opsets/opset14.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/rms_fusion.hpp>
#include <transformations/common_optimizations/shared_ops_optimization.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_avgpool_downgrade.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_convertlike.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_maxpool_downgrade.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_pad12_downgrade.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/op_conversions/convert_slice_to_strided_slice.hpp>
#include <transformations/op_conversions/convert_softmax_upgrade.hpp>
#include <transformations/op_conversions/convert_squeeze15_downgrade.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>
#include <transformations/op_conversions/detection_output_downgrade.hpp>
#include <transformations/op_conversions/einsum_decomposition.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/group_normalization_decomposition.hpp>
#include <transformations/op_conversions/group_query_attention_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>
#include <cstddef>

using namespace vpux;
using namespace IE;

namespace {

const std::string NGRAPH_ACT_SPARSITY_STATS_KEY = "activation_sparsity_statistic";
const int64_t STATIC_RANK_MAX_DIMS = 5;

class CustomOperatorAttrVisitor : public ov::AttributeVisitor {
public:
    CustomOperatorAttrVisitor(mlir::MLIRContext* ctx): _ctx(ctx) {};

    void on_adapter(const std::string& name, ov::ValueAccessor<void>&) override {
        VPUX_THROW("Attribute {0} can't be processed", name);
    }

    // The remaining adapter methods fall back on the void adapter if not implemented
    // Only string attr that is supported is "kernel_path"
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        VPUX_THROW_WHEN(name != "kernel_path", "String attribute {0} is not supported", name);
        _kernelPathAttr = mlir::StringAttr::get(_ctx, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        auto boolAttr = mlir::BoolAttr::get(_ctx, adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, boolAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int8_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getInt8Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int16_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getInt16Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getInt32Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getInt64Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<uint8_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getUInt8Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<uint16_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getUInt16Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<uint32_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getUInt32Type(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<uint64_t>& adapter) override {
        auto intAttr = mlir::IntegerAttr::get(getUInt64Type(_ctx), static_cast<int64_t>(adapter.get()));
        _attrs.emplace(_order, getNamedAttr(name, intAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) override {
        auto floatAttr = mlir::FloatAttr::get(mlir::FloatType::getF32(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, floatAttr));
        _order++;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        auto floatAttr = mlir::FloatAttr::get(mlir::FloatType::getF64(_ctx), adapter.get());
        _attrs.emplace(_order, getNamedAttr(name, floatAttr));
        _order++;
    }

public:
    // Gets final Attr Dictionary containing all registered valid Operator attributes
    // kernel_path is ommited
    mlir::DictionaryAttr getAttrDict() {
        std::vector<mlir::NamedAttribute> sortedAttrs;

        for (auto& [id, namedAttr] : _attrs) {
            // Rename by adding id prefix for to ensure DictionaryAttr preserves order
            namedAttr.setName(mlir::StringAttr::get(_ctx, to_string(id) + "_" + namedAttr.getName().str()));
            sortedAttrs.push_back(namedAttr);
        }

        return mlir::DictionaryAttr::getWithSorted(_ctx, sortedAttrs);
    }

    mlir::StringAttr getKernelPath() {
        return _kernelPathAttr;
    }

private:
    mlir::NamedAttribute getNamedAttr(const std::string& name, mlir::Attribute valAttr) {
        return mlir::NamedAttribute(mlir::StringAttr::get(_ctx, name), valAttr);
    }

    mlir::StringAttr _kernelPathAttr = nullptr;

    // Ordered map with int key to ensure ordering of attributes
    std::map<uint64_t, mlir::NamedAttribute> _attrs = {};

    mlir::MLIRContext* _ctx = nullptr;
    uint64_t _order = 0;
};

template <typename ResType>
ResType getSparsityStatsFieldChecked(const std::shared_ptr<ov::Model>& model, const std::string& primaryKey,
                                     const std::string& secondaryKey) {
    VPUX_THROW_UNLESS(model->has_rt_info(NGRAPH_ACT_SPARSITY_STATS_KEY, primaryKey, secondaryKey),
                      "Failed to query '{0}/{1}/{2}' from runtime statistics", NGRAPH_ACT_SPARSITY_STATS_KEY,
                      primaryKey, secondaryKey);
    return model->get_rt_info<ResType>(NGRAPH_ACT_SPARSITY_STATS_KEY, primaryKey, secondaryKey);
}

SmallVector<int64_t> importShape(const ov::PartialShape& shape) {
    VPUX_THROW_UNLESS(shape.rank().is_static(), "Dynamically ranked tensors are not supported");

    SmallVector<int64_t> out(checked_cast<size_t>(shape.rank().get_length()));
    std::transform(shape.begin(), shape.end(), out.begin(), [](const ov::Dimension& dim) {
        return dim.is_static() ? dim.get_length() : mlir::ShapedType::kDynamic;
    });

    return out;
}

ov::Dimension toBoundedDim(ov::Dimension dim) {
    if (dim.is_static()) {
        return dim;
    }
    auto max = dim.get_interval().get_max_val();
    VPUX_THROW_WHEN(max == ov::Interval::s_max, "Upper bounds were not specified, got the default value - '{0}'", max);
    return max;
}

template <typename T, typename InputType>
std::shared_ptr<const T> getParentNodeAs(InputType input) {
    auto sourceOutput = input.get_source_output();
    auto parentNode = sourceOutput.get_node_shared_ptr();
    return ov::as_type_ptr<const T>(parentNode);
}

}  // namespace

NGraphImporter::Callback NGraphImporter::getParser(const std::shared_ptr<ov::Node>& op) {
    using DispatchMap = std::map<ov::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::get_type_info_static(), &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ov::op::v0::Parameter::get_type_info_static(), &NGraphImporter::parseEmpty},
            {ov::op::v0::Result::get_type_info_static(), &NGraphImporter::parseEmpty},

            MAP_ENTRY(ov::opset1::Constant),
            MAP_ENTRY(ov::opset1::Convert),
            MAP_ENTRY(ov::opset1::ConvertLike),
            MAP_ENTRY(ov::opset8::Softmax),
            MAP_ENTRY(ov::opset5::LogSoftmax),
            MAP_ENTRY(ov::opset1::Tile),
            MAP_ENTRY(ov::opset1::Split),
            MAP_ENTRY(ov::opset1::Power),
            MAP_ENTRY(ov::opset1::Multiply),
            MAP_ENTRY(ov::opset1::Relu),
            MAP_ENTRY(ov::opset1::Convolution),
            MAP_ENTRY(ov::opset1::GroupConvolution),
            MAP_ENTRY(ov::opset12::GroupNormalization),
            MAP_ENTRY(ov::opset1::ConvolutionBackpropData),
            MAP_ENTRY(ov::opset1::GroupConvolutionBackpropData),
            MAP_ENTRY(ov::opset1::AvgPool),
            MAP_ENTRY(ov::opset16::AvgPool),
            MAP_ENTRY(ov::opset1::MaxPool),
            MAP_ENTRY(ov::opset8::MaxPool),
            MAP_ENTRY(ov::opset14::MaxPool),
            MAP_ENTRY(ov::opset8::AdaptiveAvgPool),
            MAP_ENTRY(ov::opset8::AdaptiveMaxPool),
            MAP_ENTRY(ov::opset1::ShuffleChannels),
            MAP_ENTRY(ov::opset8::Gather),
            MAP_ENTRY(ov::opset8::GatherND),
            MAP_ENTRY(ov::opset1::GatherTree),
            MAP_ENTRY(ov::opset8::NV12toRGB),
            MAP_ENTRY(ov::opset8::NV12toBGR),
            MAP_ENTRY(ov::opset8::I420toRGB),
            MAP_ENTRY(ov::opset8::I420toBGR),
            MAP_ENTRY(ov::opset8::RandomUniform),
            MAP_ENTRY(ov::opset1::OneHot),
            MAP_ENTRY(ov::opset5::BatchNormInference),
            MAP_ENTRY(ov::opset6::GatherElements),
            MAP_ENTRY(ov::opset4::ScatterNDUpdate),
            MAP_ENTRY(ov::opset3::ScatterUpdate),
            MAP_ENTRY(ov::opset3::ScatterElementsUpdate),
            MAP_ENTRY(ov::opset12::ScatterElementsUpdate),
            MAP_ENTRY(ov::opset1::Clamp),
            MAP_ENTRY(ov::opset1::Elu),
            MAP_ENTRY(ov::opset1::Reshape),
            MAP_ENTRY(ov::opset1::Squeeze),
            MAP_ENTRY(ov::opset1::Sigmoid),
            MAP_ENTRY(ov::opset9::GridSample),
            MAP_ENTRY(ov::opset1::LRN),
            MAP_ENTRY(ov::opset1::ReduceMax),
            MAP_ENTRY(ov::opset1::ReduceMean),
            MAP_ENTRY(ov::opset1::ReduceLogicalOr),
            MAP_ENTRY(ov::opset1::ReduceLogicalAnd),
            MAP_ENTRY(ov::opset1::ReduceProd),
            MAP_ENTRY(ov::opset1::ReduceSum),
            MAP_ENTRY(ov::opset1::ReduceMin),
            MAP_ENTRY(ov::opset4::ReduceL1),
            MAP_ENTRY(ov::opset4::ReduceL2),
            MAP_ENTRY(ov::opset1::Unsqueeze),
            MAP_ENTRY(ov::opset1::Minimum),
            MAP_ENTRY(ov::opset1::Maximum),
            MAP_ENTRY(ov::opset1::Add),
            MAP_ENTRY(ov::opset1::Divide),
            MAP_ENTRY(ov::opset1::SquaredDifference),
            MAP_ENTRY(ov::opset1::FloorMod),
            MAP_ENTRY(ov::opset1::Mod),
            MAP_ENTRY(ov::opset4::Proposal),
            MAP_ENTRY(ov::opset1::Reverse),
            MAP_ENTRY(ov::opset1::FakeQuantize),
            MAP_ENTRY(ov::op::v13::FakeConvert),
            MAP_ENTRY(ov::opset1::MatMul),
            MAP_ENTRY(ov::opset1::Tan),
            MAP_ENTRY(ov::opset1::Tanh),
            MAP_ENTRY(ov::opset1::Sin),
            MAP_ENTRY(ov::opset1::Cos),
            MAP_ENTRY(ov::opset1::Sqrt),
            MAP_ENTRY(ov::opset1::Sinh),
            MAP_ENTRY(ov::opset1::Cosh),
            MAP_ENTRY(ov::opset4::Asinh),
            MAP_ENTRY(ov::opset4::Acosh),
            MAP_ENTRY(ov::opset4::Atanh),
            MAP_ENTRY(ov::opset1::Log),
            MAP_ENTRY(ov::opset1::Selu),
            MAP_ENTRY(ov::opset2::Gelu),
            MAP_ENTRY(ov::opset1::Exp),
            MAP_ENTRY(ov::opset4::HSwish),
            MAP_ENTRY(ov::opset1::Floor),
            MAP_ENTRY(ov::opset5::Round),
            MAP_ENTRY(ov::opset4::Mish),
            MAP_ENTRY(ov::opset1::Erf),
            MAP_ENTRY(ov::opset3::Broadcast),
            MAP_ENTRY(ov::opset3::Bucketize),
            MAP_ENTRY(ov::opset1::Transpose),
            MAP_ENTRY(ov::opset4::Interpolate),
            MAP_ENTRY(ov::opset1::TopK),
            MAP_ENTRY(ov::opset3::TopK),
            MAP_ENTRY(ov::opset1::RegionYolo),
            MAP_ENTRY(ov::opset2::ReorgYolo),
            MAP_ENTRY(ov::opset1::DetectionOutput),
            MAP_ENTRY(ov::opset1::NormalizeL2),
            MAP_ENTRY(ov::opset3::CumSum),
            MAP_ENTRY(ov::opset9::Eye),
            MAP_ENTRY(ov::opset4::MVN),
            MAP_ENTRY(ov::opset6::MVN),
            MAP_ENTRY(ov::opset1::Concat),
            MAP_ENTRY(ov::opset2::ROIPooling),
            MAP_ENTRY(ov::opset1::PSROIPooling),
            MAP_ENTRY(ov::op::v9::ROIAlign),
            MAP_ENTRY(ov::opset6::ExperimentalDetectronROIFeatureExtractor),
            MAP_ENTRY(ov::opset1::StridedSlice),
            MAP_ENTRY(ov::opset1::PRelu),
            MAP_ENTRY(ov::opset4::Swish),
            MAP_ENTRY(ov::opset1::GRN),
            MAP_ENTRY(ov::opset1::Negative),
            MAP_ENTRY(ov::opset1::Sign),
            MAP_ENTRY(ov::opset1::CTCGreedyDecoder),
            MAP_ENTRY(ov::opset6::CTCGreedyDecoderSeqLen),
            MAP_ENTRY(ov::opset1::Pad),
            MAP_ENTRY(ov::opset1::LSTMCell),
            MAP_ENTRY(ov::opset4::LSTMCell),
            MAP_ENTRY(ov::opset1::Subtract),
            MAP_ENTRY(ov::opset1::LogicalAnd),
            MAP_ENTRY(ov::opset5::LSTMSequence),
            MAP_ENTRY(ov::opset1::Ceiling),
            MAP_ENTRY(ov::opset4::SoftPlus),
            MAP_ENTRY(ov::opset1::Equal),
            MAP_ENTRY(ov::opset1::Select),
            MAP_ENTRY(ov::opset9::NonMaxSuppression),
            MAP_ENTRY(ov::op::internal::NonMaxSuppressionIEInternal),
            MAP_ENTRY(ov::opset1::DepthToSpace),
            MAP_ENTRY(ov::opset1::ReverseSequence),
            MAP_ENTRY(ov::opset1::Less),
            MAP_ENTRY(ov::opset1::LessEqual),
            MAP_ENTRY(ov::opset1::NotEqual),
            MAP_ENTRY(ov::opset1::Greater),
            MAP_ENTRY(ov::opset1::GreaterEqual),
            MAP_ENTRY(ov::opset1::LogicalNot),
            MAP_ENTRY(ov::opset1::LogicalOr),
            MAP_ENTRY(ov::opset1::LogicalXor),
            MAP_ENTRY(ov::op::v13::BitwiseAnd),
            MAP_ENTRY(ov::op::v13::BitwiseOr),
            MAP_ENTRY(ov::op::v13::BitwiseXor),
            MAP_ENTRY(ov::op::v13::BitwiseNot),
            MAP_ENTRY(ov::opset1::SpaceToDepth),
            MAP_ENTRY(ov::opset2::BatchToSpace),
            MAP_ENTRY(ov::opset2::SpaceToBatch),
            MAP_ENTRY(ov::opset3::ExtractImagePatches),
            MAP_ENTRY(ov::opset1::Abs),
            MAP_ENTRY(ov::opset1::Atan),
            MAP_ENTRY(ov::opset1::Asin),
            MAP_ENTRY(ov::opset1::Acos),
            MAP_ENTRY(ov::opset7::Roll),
            MAP_ENTRY(ov::opset5::HSigmoid),
            MAP_ENTRY(ov::opset1::HardSigmoid),
            MAP_ENTRY(ov::opset3::EmbeddingBagOffsetsSum),
            MAP_ENTRY(ov::opset3::EmbeddingSegmentsSum),
            MAP_ENTRY(ov::opset3::EmbeddingBagPackedSum),
            MAP_ENTRY(ov::opset3::Assign),
            MAP_ENTRY(ov::opset3::ReadValue),
            MAP_ENTRY(ov::opset6::Assign),
            MAP_ENTRY(ov::opset6::ReadValue),
            MAP_ENTRY(ov::opset3::GRUCell),
            MAP_ENTRY(ov::opset5::GRUSequence),
            MAP_ENTRY(ov::opset1::DeformablePSROIPooling),
            MAP_ENTRY(ov::opset1::TensorIterator),
            MAP_ENTRY(ov::opset5::Loop),
            MAP_ENTRY(ov::opset7::DFT),
            MAP_ENTRY(ov::opset7::IDFT),
            MAP_ENTRY(ov::opset9::RDFT),
            MAP_ENTRY(ov::opset9::IRDFT),
            MAP_ENTRY(ov::opset8::If),
            MAP_ENTRY(ov::opset1::ShapeOf),
            MAP_ENTRY(ov::opset4::Range),
            MAP_ENTRY(ov::opset3::NonZero),
            MAP_ENTRY(ov::op::internal::RMS),
            MAP_ENTRY(ov::opset14::Inverse),
            MAP_ENTRY(ov::opset8::DeformableConvolution),
            MAP_ENTRY(ov::opset1::VariadicSplit),
            MAP_ENTRY(ov::op::internal::RoPE),
            MAP_ENTRY(ov::opset13::ScaledDotProductAttention),
    };

#undef MAP_ENTRY

    const auto dispatchIt = dispatchMap.find(op->get_type_info());

    // Default Op dispatch to OV base Op class
    auto defaultDispatch = &NGraphImporter::parseDispatch<ov::op::Op>;

    // If no match is found, dispatch to OV base Op
    return (dispatchIt != dispatchMap.end()) ? dispatchIt->second : defaultDispatch;
}

// TODO Extend implementation to check architecture, limitation and can we really compile it
bool NGraphImporter::isOpSupported(const std::shared_ptr<ov::Node>& op) {
    const bool hasParser = (NGraphImporter::getParser(op) != nullptr);
    return hasParser;
}

namespace {

mlir::Type importPrecision(mlir::MLIRContext* ctx, const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f64:
        return mlir::Float64Type::get(ctx);
    case ov::element::Type_t::f32:
        return mlir::Float32Type::get(ctx);
    case ov::element::Type_t::f16:
        return mlir::Float16Type::get(ctx);
    case ov::element::Type_t::bf16:
        return mlir::BFloat16Type::get(ctx);
    case ov::element::Type_t::f8e4m3:
        return mlir::Float8E4M3FNType::get(ctx);
    case ov::element::Type_t::f8e5m2:
        return mlir::Float8E5M2Type::get(ctx);
    case ov::element::Type_t::i64:
        return getSInt64Type(ctx);
    case ov::element::Type_t::u64:
        return getUInt64Type(ctx);
    case ov::element::Type_t::i32:
        return getSInt32Type(ctx);
    case ov::element::Type_t::u32:
        return getUInt32Type(ctx);
    case ov::element::Type_t::i16:
        return getSInt16Type(ctx);
    case ov::element::Type_t::u16:
        return getUInt16Type(ctx);
    case ov::element::Type_t::i8:
        return getSInt8Type(ctx);
    case ov::element::Type_t::u8:
        return getUInt8Type(ctx);
    case ov::element::Type_t::i4:
        return getSInt4Type(ctx);
    case ov::element::Type_t::u4:
        return getUInt4Type(ctx);
    case ov::element::Type_t::u2:
        return getUInt2Type(ctx);
    case ov::element::Type_t::boolean:
        return getBool8Type(ctx);
    case ov::element::Type_t::nf4:
        return vpux::type::QuantileFloatType::getNF4(ctx, getUInt4Type(ctx), mlir::Float16Type::get(ctx));
    default:
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

}  // namespace

//
// buildMainFunc
//

void NGraphImporter::saveInfoAboutBounds(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    auto inputs = getInputs(origNode);
    for (size_t i = 0; i < inputs.size(); i++) {
        ov::PartialShape boundedOutShape;
        auto partialShape = origNode->inputs()[i].get_partial_shape();
        if (partialShape.is_static()) {
            continue;
        }

        const auto hasBounds = [](ov::Dimension dim) {
            return dim.get_interval().has_upper_bound();
        };
        auto isBoundedShape = std::any_of(partialShape.begin(), partialShape.end(), hasBounds);

        auto ndType = mlir::cast<NDTypeInterface>(inputs[i].getType());

        if (isBoundedShape) {
            std::transform(partialShape.begin(), partialShape.end(), std::back_inserter(boundedOutShape), toBoundedDim);
            const auto boundShape = boundedOutShape.to_shape();
            const Bounds outBounds(boundShape.begin(), boundShape.end());
            if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(ndType)) {
                ndType = boundedType.changeBounds(outBounds);
            } else {
                ndType = Core::BoundedTensorType::get(ndType, outBounds);
            }
        }

        // Skip ScatterNDUpdate bounds update
        if (ov::as_type_ptr<ov::opset14::ScatterNDUpdate>(origNode) == nullptr) {
            // TODO(117112): ngraph representation is able to compute bounds more accurately than NPU compiler
            // as a workaround for StridedSlice we take shape of input data as an upper bound
            for (auto iter : origNode->input(i).get_node()->inputs()) {
                auto inputParent = iter.get_source_output().get_node_shared_ptr();
                if (auto ovStridedSlice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(inputParent)) {
                    auto inputDataShape = ovStridedSlice->get_input_tensor(0).get_partial_shape().get_max_shape();
                    const Bounds inBounds(inputDataShape.begin(), inputDataShape.end());

                    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(ndType)) {
                        ndType = boundedType.changeBounds(inBounds);
                    } else {
                        ndType = Core::BoundedTensorType::get(ndType, inBounds);
                    }
                    return;
                }
            }
        }

        // We are not able to get upper bounds for the case:
        //         *----------*
        //         |  SomeOp  |
        //         *----------*
        //              |
        //  ...    *----------*   Const
        //    \    |  ShapeOf |    /
        //     \   *----------*   /
        //      \       |        /
        //     *----------------*
        //     |     Select     | *------*
        //     | <target_shape> | | data |
        //     *----------------* *------*
        //          |               |
        //          |               |
        //          \              /
        //           \            /
        //            *-----------*
        //            | Broadcast |
        //            *-----------*
        // Since ShapeOf result is unknown at the moment, we can set an upper bound for Broadcast output
        // as a max(SomeOp: upper bounds, Const)
        if (!isBoundedShape) {
            auto getMaxValue = [](const auto& constValues, const auto& shapeValues) {
                ov::PartialShape result;
                result.reserve(constValues.size());
                std::transform(constValues.begin(), constValues.end(), shapeValues.begin(), std::back_inserter(result),
                               [](int64_t constVal, int64_t shapeVal) {
                                   return std::max(constVal, shapeVal);
                               });
                return result;
            };

            auto applyMaxValue = [](const auto& partialShape, const auto& maxValue) {
                ov::PartialShape result;
                result.reserve(partialShape.size());
                std::transform(partialShape.begin(), partialShape.end(), maxValue.begin(), std::back_inserter(result),
                               [](const auto& dim, const auto& maxVal) {
                                   return dim.is_dynamic() ? ov::Dimension(1, maxVal.get_length()) : dim;
                               });
                return result;
            };

            for (const auto& input : origNode->input(i).get_node()->inputs()) {
                auto inputParent = input.get_source_output().get_node_shared_ptr();
                auto ovBroadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(inputParent);
                if (ovBroadcast == nullptr) {
                    continue;
                }

                // See if Broadcast takes its target_shape from Select
                auto ovSelect = getParentNodeAs<ov::op::v1::Select>(ovBroadcast->input(1));
                if (ovSelect == nullptr) {
                    continue;
                }

                // Determine which input of ovSelect is ShapeOf and which is Constant
                auto shapeOfInput = getParentNodeAs<ov::opset1::ShapeOf>(ovSelect->input(1));
                auto constantInput = getParentNodeAs<ov::op::v0::Constant>(ovSelect->input(2));
                if (!shapeOfInput) {
                    shapeOfInput = getParentNodeAs<ov::opset1::ShapeOf>(ovSelect->input(2));
                    constantInput = getParentNodeAs<ov::op::v0::Constant>(ovSelect->input(1));
                }
                if (shapeOfInput == nullptr || constantInput == nullptr) {
                    continue;
                }

                // Get the values from the Constant input and the upper bound from ShapeOf
                auto constantValues = constantInput->cast_vector<int64_t>();
                auto upperBound = shapeOfInput->get_input_tensor(0).get_partial_shape().get_max_shape();

                // Calculate the maximum values from Select
                boundedOutShape = getMaxValue(constantValues, upperBound);

                auto broadcastShape = applyMaxValue(ovBroadcast->output(0).get_partial_shape(), boundedOutShape);
                ovBroadcast->set_output_type(0, ovBroadcast->output(0).get_element_type(), broadcastShape);

                if (auto dynBroadcast = mlir::dyn_cast<IE::DynamicBroadcastOp>(inputs[i].getDefiningOp())) {
                    dynBroadcast.setOutputBoundsAttr(getIntArrayAttr(builder.getContext(), boundedOutShape.to_shape()));
                }
                const auto boundedShape = boundedOutShape.to_shape();
                const Bounds bounds(boundedShape.begin(), boundedShape.end());
                ndType = Core::BoundedTensorType::get(ndType, bounds);
            }
            for (const auto& input : origNode->input(i).get_node()->inputs()) {
                auto inputParent = input.get_source_output().get_node_shared_ptr();
                auto range = ov::as_type_ptr<ov::op::v4::Range>(inputParent);
                if (range == nullptr) {
                    continue;
                }
                auto outputShape = range->get_output_partial_shape(0);
                if (outputShape.is_static()) {
                    continue;
                }

                boundedOutShape = ov::PartialShape{static_cast<int64_t>(RANGEBOUND)};
                range->set_output_type(0, range->get_output_type(), boundedOutShape);

                const auto boundedShape = boundedOutShape.to_shape();
                const auto bounds = Bounds(boundedShape.begin(), boundedShape.end());

                // Update the bounded tensor type with the new bounds
                if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(ndType)) {
                    ndType = boundedType.changeBounds(bounds);
                } else {
                    ndType = Core::BoundedTensorType::get(ndType, bounds);
                }
            }
        }
        const auto outShape = boundedOutShape.to_shape();
        const auto newTypeComponents = TypeComponents().setBounds(Bounds(outShape.begin(), outShape.end()));
        const auto changedTypeComponents = ndType.changeTypeComponents(newTypeComponents);
        inputs[i].setType(changedTypeComponents);
    }
}

mlir::func::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName,
                                                 mlir::TimingScope& rootTiming, DummyOpMode stubLayers,
                                                 bool dynamicShapeToStatic) {
    auto scopeTiming = rootTiming.nest("Import nGraph function");
    // This may be enough for now - can apply bounds taken from OV APIx
    // Simple and straightforward
    auto applyBounds = [](const ov::PartialShape& partialShape) -> ov::PartialShape {
        ov::PartialShape boundedOutShape;
        std::transform(partialShape.begin(), partialShape.end(), std::back_inserter(boundedOutShape), toBoundedDim);
        return boundedOutShape;
    };

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        auto tensorShape = param->get_partial_shape().size() == 0 ? ov::PartialShape{1} : param->get_partial_shape();
        if (dynamicShapeToStatic) {
            tensorShape = applyBounds(tensorShape);
        }
        inputTypes.push_back(importTensor(tensorShape, param->get_element_type()));
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        auto tensorShape = result->get_input_partial_shape(0).size() == 0 ? ov::PartialShape{1}
                                                                          : result->get_input_partial_shape(0);
        if (dynamicShapeToStatic) {
            tensorShape = applyBounds(tensorShape);
        }
        outputTypes.push_back(importTensor(tensorShape, result->get_input_element_type(0)));
    }

    const auto funcLoc = IE::createLayerLocation(_ctx, "main", "Func");
    const auto funcType = mlir::FunctionType::get(_ctx, ArrayRef(inputTypes), ArrayRef(outputTypes));
    auto func = moduleBuilder.create<mlir::func::FuncOp>(funcLoc, funcName, funcType);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());
        const auto paramLoc = createLayerLocation(_ctx, paramNode->get_friendly_name(), "Parameter");

        auto funcInputVal = func.getArgument(paramIndex);
        funcInputVal.setLoc(paramLoc);

        const auto partialShape = paramNode->output(0).get_partial_shape();
        if (partialShape.is_dynamic()) {
            const ov::Shape ovShape = partialShape.get_max_shape();
            const Bounds bounds(ovShape.begin(), ovShape.end());
            auto ndType = mlir::cast<NDTypeInterface>(funcInputVal.getType());
            funcInputVal.setType(Core::BoundedTensorType::get(ndType, bounds));
        }
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        const auto parser = NGraphImporter::getParser(origNode);

        if (parser != nullptr) {
            (this->*parser)(builder, origNode);
        } else {
            const auto& typeInfo = origNode->get_type_info();
            VPUX_THROW_UNLESS(stubLayers == DummyOpMode::ENABLED,
                              "[NOT IMPLEMENTED] Unsupported operation {0} with type {1} and version {2}. "
                              "Try to update the driver to the latest version. If the error persists, "
                              "please submit a bug report in https://github.com/openvinotoolkit/openvino/issues",
                              origNode->get_friendly_name(), typeInfo.name, typeInfo.version_id);
            parseNodeAsStub(builder, origNode);
        }
        if (!dynamicShapeToStatic) {
            saveInfoAboutBounds(builder, origNode);
        }
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(_netGraph->get_results().size());

    for (const auto& p : _netGraph->get_results() | indexed) {
        const auto& resultNode = p.value();

        _log.trace("Convert network Result {0}", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        funcOutputs.push_back(resultInputs[0]);
    }

    const auto retLoc = createLayerLocation(_ctx, "output", "Output");
    auto returnOp = builder.create<mlir::func::ReturnOp>(retLoc, ArrayRef(funcOutputs));

    _log.trace("Inserted func.return at {0}", returnOp->getLoc());

    if (!dynamicShapeToStatic) {
        const auto parentBlock = returnOp->getOperand(0).getParentBlock();
        // Set bounds for function parameters
        VPUX_THROW_WHEN(inputTypes.size() != parentBlock->getNumArguments(),
                        "The number of input types {0} does not match the number of function arguments {1}",
                        inputTypes.size(), parentBlock->getNumArguments());
        for (const auto& argIdx : irange(inputTypes.size())) {
            inputTypes[argIdx] = parentBlock->getArgument(argIdx).getType();
        }

        // Set bounds for function results
        VPUX_THROW_WHEN(outputTypes.size() != funcOutputs.size(),
                        "The number of input types {0} does not match the number of function results {1}",
                        inputTypes.size(), funcOutputs.size());
        for (const auto& funcOutType : funcOutputs | indexed) {
            outputTypes[funcOutType.index()] = funcOutType.value().getType();
        }

        if (updateFunctionSignature(func, inputTypes, outputTypes, _log).failed()) {
            _log.debug("Failed to updateFunctionSignature");
            return func;
        }
    }

    return func;
}

void NGraphImporter::buildBlockFromRegion(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }
    SmallVector<mlir::Location> inputTypeLocations(inputTypes.size(), loc);

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }
    SmallVector<mlir::Location> outputTypeLocations(outputTypes.size(), loc);

    block->addArguments(inputTypes, inputTypeLocations);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = block->getArgument(paramIndex);
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        const auto parser = NGraphImporter::getParser(origNode);

        if (parser != nullptr) {
            (this->*parser)(builder, origNode);
        } else {
            parseNodeAsStub(builder, origNode);
        }
    }
    SmallVector<mlir::Value> blockOutputs;
    blockOutputs.reserve(_netGraph->get_results().size());
    for (const auto& p : _netGraph->get_results() | indexed) {
        const auto& resultNode = p.value();

        _log.trace("Convert network Result {0}", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        blockOutputs.push_back(resultInputs[0]);
    }
    builder.create<IE::YieldOp>(loc, blockOutputs);
}

void NGraphImporter::buildBlockFromBody(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(_netGraph->get_parameters().size());
    for (const auto& param : _netGraph->get_parameters()) {
        inputTypes.push_back(importTensor(param->get_partial_shape(), param->get_element_type()));
    }
    SmallVector<mlir::Location> inputTypeLocations(inputTypes.size(), loc);

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }
    SmallVector<mlir::Location> outputTypeLocations(outputTypes.size(), loc);

    block->addArguments(inputTypes, inputTypeLocations);

    for (const auto& p : _netGraph->get_parameters() | indexed) {
        const auto& paramNode = p.value();
        const auto paramIndex = checked_cast<uint32_t>(p.index());

        _log.trace("Convert network Parameter {0}", paramNode->get_friendly_name());

        const auto funcInputVal = block->getArgument(paramIndex);
        _importedVals.emplace(paramNode->output(0), funcInputVal);
    }

    for (const auto& origNode : _netGraph->get_ordered_ops()) {
        const auto parser = NGraphImporter::getParser(origNode);

        // Add tensor iterator loc as prefix for node name to make node loc unique
        auto origName = origNode->get_friendly_name();
        origNode->set_friendly_name(stringifyPrimaryLocation(loc) + "/" + origName);
        if (parser != nullptr) {
            (this->*parser)(builder, origNode);
        } else {
            const auto& typeInfo = origNode->get_type_info();
            VPUX_THROW("[NOT IMPLEMENTED] Unsupported operation {0} with type {1} and version {2}. "
                       "Try to update the driver to the latest version. If the error persists, "
                       "please submit a bug report in https://github.com/openvinotoolkit/openvino/issues",
                       origNode->get_friendly_name(), typeInfo.name, typeInfo.version_id);
            parseNodeAsStub(builder, origNode);
        }
        _log.trace("Convert {0} layer {1}", origNode->get_type_name(), origNode->get_friendly_name());
        // Here restore the node name to generate clean names.
        // In some cases, 2 tensor iterator node share the same _netGraph.
        origNode->set_friendly_name(origName);
    }
    SmallVector<mlir::Value> blockOutputs;
    blockOutputs.reserve(_netGraph->get_results().size());
    for (const auto& p : _netGraph->get_results() | indexed) {
        const auto& resultNode = p.value();
        _log.trace("Convert network Result {0} into LoopTerminator.", resultNode->get_friendly_name());

        const auto resultInputs = getInputs(resultNode);
        VPUX_THROW_UNLESS(resultInputs.size() == 1, "nGraph Result {0} has unsupported number of inputs {1}",
                          resultNode->get_friendly_name(), resultInputs.size());

        blockOutputs.push_back(resultInputs[0]);
    }
    builder.create<IE::LoopTerminatorOp>(loc, blockOutputs);
}

SmallVector<mlir::Type> NGraphImporter::getRegionResults() {
    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(_netGraph->get_results().size());
    for (const auto& result : _netGraph->get_results()) {
        outputTypes.push_back(importTensor(result->get_input_partial_shape(0), result->get_input_element_type(0)));
    }
    return outputTypes;
}

SmallVector<mlir::Type> NGraphImporter::getLoopLikeRegionResults(int64_t numIter, int32_t numResults,
                                                                 ArrayRef<mlir::Attribute> concatOutputVector,
                                                                 ArrayRef<mlir::Attribute> invariantOutputVector) {
    mlir::DenseMap<int64_t, int64_t> toConcatOutputAxisMap, toConcatOutputPortIdMap, invariantOutputPortIdMap;
    for (const auto attr : concatOutputVector) {
        auto copm = mlir::dyn_cast_or_null<vpux::IE::ConcatOutputPortMapAttr>(attr);
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto inId = copm.getInternalLayerId().getValue().getSExtValue();
        auto axis = copm.getAxis().getValue().getSExtValue();
        toConcatOutputAxisMap.insert({inId, axis});
        toConcatOutputPortIdMap.insert({exId, inId});
    }

    for (const auto attr : invariantOutputVector) {
        auto iopm = mlir::dyn_cast_or_null<vpux::IE::InvariantOutputPortMapAttr>(attr);
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto inId = iopm.getInternalLayerId().getValue().getSExtValue();
        invariantOutputPortIdMap.insert({exId, inId});
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(numResults);
    // Loop Terminator has 1 more output(execution condition output) than Loop Op output
    for (auto exId : irange(numResults)) {
        auto inId =
                toConcatOutputPortIdMap.count(exId) ? toConcatOutputPortIdMap[exId] : invariantOutputPortIdMap[exId];
        std::shared_ptr<ov::op::v0::Result> result = _netGraph->get_results()[inId];
        ov::Shape targetShape = result->get_input_partial_shape(0).get_max_shape();

        if (toConcatOutputPortIdMap.count(exId)) {
            int64_t axis = toConcatOutputAxisMap[inId] < 0 ? toConcatOutputAxisMap[inId] + targetShape.size()
                                                           : toConcatOutputAxisMap[inId];
            VPUX_THROW_UNLESS(axis >= 0, "Wrong axis `{0}`, out of range", toConcatOutputAxisMap[inId]);
            targetShape[axis] = numIter * targetShape[axis];
        }

        outputTypes.push_back(importTensor(targetShape, result->get_input_element_type(0)));
    }
    return outputTypes;
}

//
// Parsers
//

void NGraphImporter::parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    _log.debug("{0} is not supported. Replacing with a Stub layer", origNode->get_friendly_name());

    const auto inputs = getInputs(origNode);
    const auto outputCount = static_cast<int32_t>(origNode->get_output_size());

    SmallVector<mlir::Type> outputTypes;
    for (auto i = 0; i < outputCount; ++i) {
        auto type = importTensor(origNode->get_output_shape(i), origNode->get_output_element_type(i));
        outputTypes.push_back(type);
    }

    auto op = builder.create<IE::StubOp>(createLocation(origNode), outputTypes, inputs);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::BatchToSpace>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::BatchToSpace>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph BatchToSpace node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::BatchToSpace>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                               nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Constant>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Constant>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.empty(), "nGraph Constant node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto tensorType = importTensor(origNode->get_output_partial_shape(0), origNode->get_output_element_type(0));
    const Bit elemTypeSize = getElemTypeSize(tensorType);
    const auto bitWidth = elemTypeSize.count();
    const auto bufferSize = static_cast<size_t>(vpux::getExpectedBufferSize(tensorType).count());
    // the assumption here is that data is byte-aligned. additionally, if the
    // OV's byte size doesn't match the expectation, it is likely a bug.
    VPUX_THROW_WHEN(bufferSize != origNode->get_byte_size(),
                    "Unexpected OV constant buffer size. Expected {0} bytes, actual {1} bytes", bufferSize,
                    origNode->get_byte_size());

    auto value = [&]() -> mlir::ElementsAttr {
        const auto rawBuffer = ArrayRef(origNode->get_data_ptr<char>(), bufferSize);
        // DenseElementsAttr has very limited support for sub byte type (only I1 is supported).
        // Therefore, we need to avoid using DenseElementsAttr to store sub byte type.

        if (!vpux::Const::isSubByte(bitWidth) && !_sharedConstants) {
            return mlir::DenseElementsAttr::getFromRawBuffer(tensorType, rawBuffer);
        }

        std::string cstName;
        ov::RTMap& runtimeInfoMap = origNode->get_rt_info();

        cstName = [&]() -> std::string {
            const auto& weightlessCacheAttrIt =
                    runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                if (origNode->get_element_type() == weightlessCacheAttr.original_dtype) {
                    return formatv("{0}{1}", vpux::Const::IMPORTED_WEIGHT_PREFIX, weightlessCacheAttr.bin_offset).str();
                } else {
                    Logger::global().debug(
                            "Weightless cache attribute type {0} does not match constant type {1} for node '{2}'",
                            weightlessCacheAttr.original_dtype, origNode->get_element_type(),
                            origNode->get_friendly_name());
                }
            }

            return "INTERNAL_CONSTANT";
        }();

        return Const::createExternalConstContent(tensorType, rawBuffer, cstName);
    }();

    Const::ContentSetup contentSetup(value.getType());

    auto dataType = importPrecision(_ctx, origNode->get_output_element_type(0));
    if (vpux::Const::isSubByte(bitWidth) || mlir::isa<vpux::type::QuantileFloatType>(dataType)) {
        mlir::Type conversionType =
                dataType.isSignedInteger() ? getSInt8Type(builder.getContext()) : getUInt8Type(builder.getContext());
        contentSetup = contentSetup.convertElemType(conversionType).castElemType(dataType);
        tensorType = mlir::RankedTensorType::get(tensorType.getShape(), dataType);
    }

    auto op = builder.create<Const::DeclareOp>(createLocation(origNode), tensorType,
                                               Const::ContentAttr::get(value, contentSetup));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convert>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Convert>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Convert node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstType = importPrecision(_ctx, origNode->get_destination_type());
    const auto dstTypeAttr = mlir::TypeAttr::get(dstType);

    auto op = builder.create<IE::ConvertOp>(createLocation(origNode), inputs[0], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ConvertLike>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ConvertLike>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ConvertLike node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ConvertLikeOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::Softmax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Softmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::SoftMaxOp>(createLocation(origNode), inputs[0], axisAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LogSoftmax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::LogSoftmax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph LogSoftmax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::LogSoftmaxOp>(createLocation(origNode), inputs[0], axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tile>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tile>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Tile node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::TileOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Relu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Relu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReLUOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Split>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Split>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Split node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto num_splits = origNode->get_num_splits();
    const auto numSplitsAttr = getIntAttr(_ctx, num_splits);

    auto op = builder.create<IE::SplitOp>(createLocation(origNode), inputs[0], inputs[1], numSplitsAttr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Power>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Power>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Power node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::PowerOp>(createLocation(origNode), inputs[0], inputs[1],
                                          importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Multiply>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Multiply>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Multiply node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MultiplyOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type), nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MatMul>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::MatMul>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MatMulOp>(createLocation(origNode), inputs[0], inputs[1], origNode->get_transpose_a(),
                                           origNode->get_transpose_b(), nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Convolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Convolution>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::ConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, attrStride,
                                                attrPadsBegin, attrPadsEnd, attrDilation, nullptr, nullptr, nullptr,
                                                nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::GroupConvolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GroupConvolution>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());

    auto op = builder.create<IE::GroupConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                                     attrStride, attrPadsBegin, attrPadsEnd, attrDilation,
                                                     /*groups=*/nullptr,
                                                     /*post_op=*/nullptr,
                                                     /*clamp=*/nullptr,
                                                     /*outputPadding=*/nullptr,
                                                     /*inputPadding=*/nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset12::GroupNormalization>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v12::GroupNormalization>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrNumGroups = builder.getI32IntegerAttr(static_cast<int32_t>(origNode->get_num_groups()));
    const auto attrEpsilon = builder.getF32FloatAttr(origNode->get_epsilon());

    auto op = builder.create<IE::GroupNormalizationOp>(createLocation(origNode),
                                                       inputs[0],  // input
                                                       inputs[1],  // scale
                                                       inputs[2],  // bias
                                                       attrNumGroups, attrEpsilon);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::ConvolutionBackpropData>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ConvolutionBackpropData>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2) || (inputs.size() == 3),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getIntArrayAttr(_ctx, origNode->get_output_padding());

    auto optionalOutputShapeInput = inputs.size() == 2 ? nullptr : inputs[2];
    auto op = builder.create<IE::ConvolutionBackpropDataOp>(createLocation(origNode), inputs[0], inputs[1],
                                                            optionalOutputShapeInput, attrStride, attrPadsBegin,
                                                            attrPadsEnd, attrDilation, attrOutputPadding);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::GroupConvolutionBackpropData>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GroupConvolutionBackpropData>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2) || (inputs.size() == 3),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrOutputPadding = getIntArrayAttr(_ctx, origNode->get_output_padding());

    auto optionalOutputShapeInput = inputs.size() == 2 ? nullptr : inputs[2];
    auto op = builder.create<IE::GroupConvolutionBackpropDataOp>(createLocation(origNode), inputs[0], inputs[1],
                                                                 optionalOutputShapeInput, attrStride, attrPadsBegin,
                                                                 attrPadsEnd, attrDilation, attrOutputPadding);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::AvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::AvgPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());
    const auto attrExcludePads = origNode->get_exclude_pad() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::AvgPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, attrExcludePads, nullptr,
                                            nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset16::AvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v16::AvgPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrDilations = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());
    const auto attrExcludePads = origNode->get_exclude_pad() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::AvgPool16Op>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                              attrDilations, attrPadsBegin, attrPadsEnd, attrRoundingType,
                                              attrExcludePads, nullptr, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::MaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    auto op = builder.create<IE::MaxPoolOp>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                            attrPadsBegin, attrPadsEnd, attrRoundingType, nullptr, nullptr, nullptr,
                                            nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::MaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto dstTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));
    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());

    auto op = builder.create<IE::MaxPool8Op>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                             attrDilation, attrPadsBegin, attrPadsEnd, attrRoundingType, dstTypeAttr,
                                             axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::MaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v14::MaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto attrKernelSize = getIntArrayAttr(_ctx, origNode->get_kernel());
    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());

    const auto dstTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));
    const auto attrRoundingType = importRoundingType(origNode->get_rounding_type());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());

    auto op = builder.create<IE::MaxPool8Op>(createLocation(origNode), inputs[0], attrKernelSize, attrStride,
                                             attrDilation, attrPadsBegin, attrPadsEnd, attrRoundingType, dstTypeAttr,
                                             axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::AdaptiveAvgPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AdaptiveAvgPoolOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::AdaptiveMaxPool>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto dstTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::AdaptiveMaxPoolOp>(createLocation(origNode), inputs[0], inputs[1], dstTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Add>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Add>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AddOp>(
            createLocation(origNode), inputs[0], inputs[1], importBroadcastType(autob.m_type),
            /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding*/ nullptr, /*inputPadding*/ nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Divide>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Divide>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::DivideOp>(createLocation(origNode), inputs[0], inputs[1],
                                           importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::SquaredDifference>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::SquaredDifference>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SquaredDifferenceOp>(createLocation(origNode), inputs[0], inputs[1],
                                                      importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::FloorMod>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::FloorMod>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::FloorModOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Mod>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Mod>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::ModOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShuffleChannels>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ShuffleChannels>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ShuffleChannels node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ShuffleChannelsOp>(createLocation(origNode), inputs[0], origNode->get_axis(),
                                                    origNode->get_group());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::Gather>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Gather node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto batchDims = origNode->get_batch_dims();
    auto idxRank = origNode->get_input_partial_shape(1).rank().get_length();
    batchDims = (batchDims < 0) ? (batchDims + idxRank) : batchDims;
    auto normBatchDims = getIntAttr(_ctx, batchDims);
    VPUX_THROW_UNLESS(batchDims >= 0, "Invalid batch_dims value '{0}'", batchDims);

    auto op = builder.create<IE::GatherOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                           normBatchDims, getIntAttr(_ctx, idxRank));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::GatherND>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GatherND node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto batchDims = origNode->get_batch_dims();

    auto op =
            builder.create<IE::GatherNDOp>(createLocation(origNode), inputs[0], inputs[1], getIntAttr(_ctx, batchDims));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GatherTree>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::GatherTree>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph GatherTree node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GatherTreeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::NV12toRGB>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && !inputs.empty(),
                      "nGraph NV12toRGB node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, nullptr,
                                             IE::ColorFmt::NV12, IE::ColorFmt::RGB);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::NV12toBGR>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && !inputs.empty(),
                      "nGraph NV12toBGR node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, nullptr,
                                             IE::ColorFmt::NV12, IE::ColorFmt::BGR);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::I420toRGB>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 1) || (inputs.size() == 3),
                      "nGraph I420toRGB node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto thirdInput = inputs.size() == 1 ? nullptr : inputs[2];
    IE::YuvToRgbOp op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, thirdInput,
                                                       IE::ColorFmt::I420, IE::ColorFmt::RGB);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::I420toBGR>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 1) || (inputs.size() == 3),
                      "nGraph I420toBGR node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
    auto thirdInput = inputs.size() == 1 ? nullptr : inputs[2];
    IE::YuvToRgbOp op = builder.create<IE::YuvToRgbOp>(createLocation(origNode), inputs[0], secondInput, thirdInput,
                                                       IE::ColorFmt::I420, IE::ColorFmt::BGR);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::RandomUniform>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph RandomUniform node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto shapeConstant = dynamic_cast<ov::opset7::Constant*>(origNode->input_value(0).get_node());
    VPUX_THROW_UNLESS(shapeConstant != nullptr,
                      "nGraph RandomUniform node '{0}' must have Constant shape input in order to infer output shape.",
                      origNode->get_friendly_name());
    const auto shapeValues = shapeConstant->cast_vector<int64_t>();
    const auto shapeAttr = getIntArrayAttr(_ctx, shapeValues);

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_input_element_type(1)));
    const auto globalSeedAttr = getIntAttr(_ctx, origNode->get_global_seed());
    const auto opSeedAttr = getIntAttr(_ctx, origNode->get_op_seed());

    auto op = builder.create<IE::RandomUniformOp>(createLocation(origNode), inputs[1], inputs[2], shapeAttr,
                                                  outputTypeAttr, globalSeedAttr, opSeedAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset12::OneHot>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::OneHot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph OneHot node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto outElemTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_element_type()));

    IE::OneHotOp op = builder.create<IE::OneHotOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                   nullptr, nullptr, nullptr, axisAttr, outElemTypeAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset5::BatchNormInference>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::BatchNormInference>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph BatchNorm node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto epsAttr = getFPAttr(_ctx, origNode->get_eps_value());
    auto op =
            builder.create<IE::BatchNormInferenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                     inputs[3], inputs[4], nullptr, nullptr, nullptr, nullptr, epsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::GatherElements>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::GatherElements>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GatherElements node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto origAxis = origNode->get_axis();
    auto rank = origNode->get_input_partial_shape(0).rank().get_length();
    origAxis = (origAxis < 0) ? (origAxis + rank) : origAxis;
    auto axis = getIntAttr(_ctx, origAxis);

    auto op = builder.create<IE::GatherElementsOp>(createLocation(origNode), inputs[0], inputs[1], axis);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ScatterNDUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset4::ScatterNDUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ScatterNDUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterNDUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ScatterUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ScatterUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph ScatterUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ScatterUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                  nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset12::ScatterElementsUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v12::ScatterElementsUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4,
                      "nGraph ScatterElementsUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto reductionTypeAttr = importScatterElementsUpdateReductionType(origNode->get_reduction());
    const auto useInitValAttr = mlir::BoolAttr::get(_ctx, origNode->get_use_init_val());
    auto op = builder.create<IE::ScatterElementsUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                          inputs[3], nullptr, reductionTypeAttr, useInitValAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::ScatterElementsUpdate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ScatterElementsUpdate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4,
                      "nGraph ScatterElementsUpdate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto defaultReductionTypeAttr =
            importScatterElementsUpdateReductionType(ov::op::v12::ScatterElementsUpdate::Reduction::NONE);
    auto useInitValAttr = mlir::BoolAttr::get(_ctx, true);
    auto op = builder.create<IE::ScatterElementsUpdateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                          inputs[3], nullptr, defaultReductionTypeAttr, useInitValAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reshape>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Reshape>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Reshape node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    if (origNode->output(0).get_partial_shape().is_static()) {
        auto op = builder.create<IE::ReshapeOp>(createLocation(origNode), inputs[0], inputs[1],
                                                origNode->get_special_zero(), nullptr);
        addOutputs(origNode, op);
    } else {
        const auto outputShape = importShape(origNode->output(0).get_partial_shape());
        const auto outputShapeAttr = getIntArrayAttr(builder.getContext(), outputShape);
        const auto outputBounds = origNode->output(0).get_partial_shape().get_max_shape();
        const auto outputBoundsAttr = getIntArrayAttr(builder.getContext(), outputBounds);
        auto op = builder.create<IE::DynamicReshapeOp>(createLocation(origNode), inputs[0], inputs[1], outputShapeAttr,
                                                       outputBoundsAttr);
        addOutputs(origNode, op);
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Minimum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Minimum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Minimum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MinimumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Maximum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Maximum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Maximum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::MaximumOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Clamp>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Clamp>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Clamp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto min = origNode->get_min();
    const auto max = origNode->get_max();
    const auto minAttr = getFPAttr(_ctx, min);
    const auto maxAttr = getFPAttr(_ctx, max);

    auto op = builder.create<IE::ClampOp>(createLocation(origNode), inputs[0], minAttr, maxAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Proposal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Proposal>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Proposal node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& proposalParam = origNode->get_attrs();
    const auto proposalParamAttr = importProposalAttrs(proposalParam);

    auto op = builder.create<IE::ProposalOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                             proposalParamAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Reverse>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Reverse>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Reverse node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ReverseOp>(createLocation(origNode), inputs[0], inputs[1], nullptr,
                                            importReverseMode(origNode->get_mode()));

    addOutputs(origNode, op);
}

bool checkDynamicInputsOutputs(const std::shared_ptr<ov::Node>& origNode) {
    const auto isDynamic = [](auto node) {
        return node.get_partial_shape().is_dynamic();
    };

    const auto hasDynamicInputs = llvm::any_of(origNode->inputs(), isDynamic);
    const auto hasDynamicOutputs = llvm::any_of(origNode->outputs(), isDynamic);

    return hasDynamicInputs || hasDynamicOutputs;
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Unsqueeze>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Unsqueeze>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::UnsqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::LRN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto beta = origNode->get_beta();
    const auto bias = origNode->get_bias();
    const auto size = origNode->get_nsize();

    const auto alphaAttr = getFPAttr(_ctx, alpha);
    const auto betaAttr = getFPAttr(_ctx, beta);
    const auto biasAttr = getFPAttr(_ctx, bias);
    const auto sizeAttr = getIntAttr(_ctx, size);

    auto op = builder.create<IE::LRNOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, alphaAttr, betaAttr,
                                        biasAttr, sizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Broadcast>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Broadcast>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph Broadcast node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto mode = importBroadcastMode(origNode->get_broadcast_spec().m_type);

    if (checkDynamicInputsOutputs(origNode)) {
        const auto outputShape = importShape(origNode->output(0).get_partial_shape());
        const auto outputShapeAttr = getIntArrayAttr(builder.getContext(), outputShape);
        const auto outputBounds = origNode->output(0).get_partial_shape().get_max_shape();
        const auto outputBoundsAttr = getIntArrayAttr(builder.getContext(), outputBounds);

        auto thirdInput = inputs.size() >= 3 ? inputs[2] : nullptr;
        IE::DynamicBroadcastOp op = builder.create<IE::DynamicBroadcastOp>(
                createLocation(origNode), inputs[0], inputs[1], thirdInput, mode, outputShapeAttr, outputBoundsAttr);
        addOutputs(origNode, op);
    } else {
        auto thirdInput = inputs.size() >= 3 ? inputs[2] : nullptr;
        IE::BroadcastOp op =
                builder.create<IE::BroadcastOp>(createLocation(origNode), inputs[0], inputs[1], thirdInput, mode);
        addOutputs(origNode, op);
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Bucketize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Bucketize>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Bucketize node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_output_type()));
    const auto withRightBoundAttr = origNode->get_with_right_bound() ? mlir::UnitAttr::get(_ctx) : nullptr;

    VPUX_THROW_UNLESS(outputTypeAttr == mlir::TypeAttr::get(importPrecision(_ctx, ov::element::i32)),
                      "nGraph Bucketize output_type attribute '{0}' should have only SI32 type", outputTypeAttr);

    auto op = builder.create<IE::BucketizeOp>(createLocation(origNode), inputs[0], inputs[1], outputTypeAttr,
                                              withRightBoundAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMax>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMax>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMax node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMaxOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMean>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMean>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMean node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMeanOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims,
                                               nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceLogicalOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceLogicalOr>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceLogicalOr node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceLogicalOrOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::ReduceLogicalAnd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceLogicalAnd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceLogicalAnd node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op =
            builder.create<IE::ReduceLogicalAndOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceProd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceProd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceProd node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceProdOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceSumOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims,
                                              nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReduceMin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::ReduceMin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceMin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceMinOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL1>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::ReduceL1>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceL1 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceL1Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::ReduceL2>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::ReduceL2>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReduceL2 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto keep_dims = origNode->get_keep_dims();

    auto op = builder.create<IE::ReduceL2Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, keep_dims);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SigmoidOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::GridSample>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    const auto alignCornersAttr = origNode->get_attributes().align_corners ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto modeAttr = importGridSampleMode(origNode->get_attributes().mode);
    const auto paddingModeAttr = importGridSamplePaddingMode(origNode->get_attributes().padding_mode);

    auto op = builder.create<IE::GridSampleOp>(createLocation(origNode), inputs[0], inputs[1], alignCornersAttr,
                                               modeAttr, paddingModeAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Squeeze>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Squeeze>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 2 && inputs.size() > 0,
                      "nGraph Squeeze node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    if (checkDynamicInputsOutputs(origNode)) {
        const auto outputShape = importShape(origNode->output(0).get_partial_shape());
        const auto outputShapeAttr = getIntArrayAttr(builder.getContext(), outputShape);
        const auto outputBounds = origNode->output(0).get_partial_shape().get_max_shape();
        const auto outputBoundsAttr = getIntArrayAttr(builder.getContext(), outputBounds);

        const auto outShapeRank = checked_cast<int64_t>(outputShape.size());
        const auto dataType = mlir::RankedTensorType::get({outShapeRank}, getSInt32Type(builder.getContext()));
        auto outputShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(outputShape), -1);

        const auto shapeTensor =
                Const::createConst(builder, createLocation(origNode), dataType, ArrayRef(outputShapeValues));
        auto op = builder.create<IE::DynamicReshapeOp>(createLocation(origNode), inputs[0], shapeTensor,
                                                       outputShapeAttr, outputBoundsAttr);
        addOutputs(origNode, op);
    } else {
        if (inputs.size() == 2) {
            auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
            addOutputs(origNode, op);
        } else {
            auto op = builder.create<IE::SqueezeOp>(createLocation(origNode), inputs[0], nullptr, nullptr);
            addOutputs(origNode, op);
        }
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Transpose>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Transpose>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Transpose node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TransposeOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tan>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tan>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tan node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Tanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Tanh>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Tanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::TanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SinOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Cos>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Cos>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Cos node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CosOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sqrt>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sqrt>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sqrt node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SqrtOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Cosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Cosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Cosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Asinh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Asinh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asinh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Acosh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Acosh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acosh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcoshOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Atanh>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Atanh>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atanh node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanhOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Roll>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::Roll>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Roll node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RollOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Log>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Log>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Log node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Selu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Selu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Selu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SeluOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr, nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Gelu>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Gelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::GeluOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Elu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Elu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Elu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto alpha = origNode->get_alpha();
    const auto alphaAttr = getFPAttr(_ctx, alpha);

    auto op = builder.create<IE::EluOp>(createLocation(origNode), inputs[0], alphaAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::HSwish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::HSwish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSwish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSwishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Floor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Floor>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Floor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::FloorOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Round>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::Round>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Round node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RoundOp>(createLocation(origNode), inputs[0], importRoundMode(origNode->get_mode()));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Mish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Mish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Mish node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::MishOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Erf>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Erf>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Erf node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ErfOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::FakeQuantize>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::FakeQuantize>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 5, "nGraph FakeQuantize node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_auto_broadcast();

    const auto levelsAttr = getIntAttr(_ctx, origNode->get_levels());

    // ov FakeQunatize does not support fp8, nullptr is passed.
    auto op = builder.create<IE::FakeQuantizeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 inputs[4], levelsAttr, /*lowFpType=*/nullptr,
                                                 importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::FakeConvert>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v13::FakeConvert>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 2 || inputs.size() == 3),
                      "nGraph FakeConvert node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    const auto& destinationTypeName = origNode->get_destination_type();

    const auto destinationTypeAttr = [&] {
        if (destinationTypeName == "f8e4m3") {
            return mlir::TypeAttr::get(mlir::Float8E4M3FNType::get(_ctx));
        } else if (destinationTypeName == "f8e5m2") {
            return mlir::TypeAttr::get(mlir::Float8E5M2Type::get(_ctx));
        }
        return mlir::TypeAttr::get(mlir::NoneType::get(_ctx));
    }();

    VPUX_THROW_WHEN(mlir::isa<mlir::NoneType>(destinationTypeAttr.getValue()),
                    "nGraph FakeConvert unsupported destination type: '{0}'.", destinationTypeName);

    const auto data = inputs[0];
    const auto scale = inputs[1];
    const auto shift = (inputs.size() == 3) ? inputs[2] : nullptr;
    const auto op =
            builder.create<IE::FakeConvertOp>(createLocation(origNode), data, scale, shift, destinationTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Exp>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Exp>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Exp node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::ExpOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::StridedSlice>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::StridedSlice>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph StridedSlice node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto attrBeginMask = getIntArrayAttr(_ctx, origNode->get_begin_mask());
    auto attrEndMask = getIntArrayAttr(_ctx, origNode->get_end_mask());
    auto attrNewAxisMask = getIntArrayAttr(_ctx, origNode->get_new_axis_mask());
    auto attrShrinkAxisMask = getIntArrayAttr(_ctx, origNode->get_shrink_axis_mask());
    auto attrEllipsisAxisMask = getIntArrayAttr(_ctx, origNode->get_ellipsis_mask());

    auto op = builder.create<IE::StridedSliceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                 nullptr, nullptr, nullptr, attrBeginMask, attrEndMask, attrNewAxisMask,
                                                 attrShrinkAxisMask, attrEllipsisAxisMask);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputSize = getIntArrayAttr(_ctx, origNode->get_output_roi());
    const auto spatialScaleAttr = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto method = importROIPoolingMethod(origNode->get_method());

    auto op = builder.create<IE::ROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1], outputSize,
                                               spatialScaleAttr, method);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::ROIAlign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph ROIAlign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto pooled_h = getIntAttr(_ctx, origNode->get_pooled_h());
    const auto pooled_w = getIntAttr(_ctx, origNode->get_pooled_w());
    const auto sampling_ratio = getIntAttr(_ctx, origNode->get_sampling_ratio());
    const auto spatialScaleAttr = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto poolingMode = importROIAlignMethod(origNode->get_mode());
    const auto alignedMode = importROIAlignAlignedMethod(origNode->get_aligned_mode());

    auto op = builder.create<IE::ROIAlignOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], pooled_h,
                                             pooled_w, sampling_ratio, spatialScaleAttr, poolingMode, alignedMode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset6::ExperimentalDetectronROIFeatureExtractor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type,
                               ov::op::v6::ExperimentalDetectronROIFeatureExtractor>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(
            (inputs.size() == 2) || (inputs.size() == 3) || (inputs.size() == 4),
            "nGraph ExperimentalDetectronROIFeatureExtractor node '{0}' has unsupported number of inputs '{1}'",
            origNode->get_friendly_name(), inputs.size());

    const auto expDetectronROIFeatureExtractAttr = importExpDetectronROIFeatureExtractAttrs(origNode->get_attrs());

    auto op = builder.create<IE::ExperimentalDetectronROIFeatureExtractorOp>(createLocation(origNode), inputs,
                                                                             expDetectronROIFeatureExtractAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Concat>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Concat>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() >= 1, "nGraph Concat node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axis = origNode->get_axis();
    const auto axisAttr = getIntAttr(_ctx, axis);

    auto op = builder.create<IE::ConcatOp>(createLocation(origNode), inputs, axisAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Interpolate>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Interpolate>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 4) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto interpolateAttr = importInterpolateAttrs(origNode->get_attrs());

    IE::InterpolateOp op;
    if (inputs.size() == 3) {
        op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, interpolateAttr,
                                               nullptr, nullptr);
    } else {
        op = builder.create<IE::InterpolateOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, interpolateAttr,
                                               nullptr, nullptr);
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v3::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::TopK>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto modeAttr = importTopKMode(origNode->get_mode());
    const auto sortTypeAttr = importTopKSortType(origNode->get_sort_type());
    const auto indexElementTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::TopKOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, axisAttr, modeAttr,
                                         sortTypeAttr, indexElementTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v1::TopK>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::TopK>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph TopK node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto modeAttr = importTopKMode(origNode->get_mode());
    const auto sortTypeAttr = importTopKSortType(origNode->get_sort_type());
    const auto indexElementTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_index_element_type()));

    auto op = builder.create<IE::TopKOp>(createLocation(origNode), inputs[0], inputs[1], nullptr, axisAttr, modeAttr,
                                         sortTypeAttr, indexElementTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::RegionYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::RegionYolo>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph RegionYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto coordAttr = getIntAttr(_ctx, origNode->get_num_coords());
    const auto classesAttr = getIntAttr(_ctx, origNode->get_num_classes());
    const auto regionsAttr = getIntAttr(_ctx, origNode->get_num_regions());
    const auto doSoftmaxAttr = mlir::BoolAttr::get(_ctx, origNode->get_do_softmax());
    const auto maskAttr = getIntArrayAttr(_ctx, origNode->get_mask());
    const auto axisAttr = getIntAttr(_ctx, origNode->get_axis());
    const auto axisEndAttr = getIntAttr(_ctx, origNode->get_end_axis());
    const auto anchorsAttr = getFPArrayAttr(_ctx, origNode->get_anchors());

    auto op = builder.create<IE::RegionYoloOp>(createLocation(origNode), inputs[0], coordAttr, classesAttr, regionsAttr,
                                               doSoftmaxAttr, maskAttr, axisAttr, axisEndAttr, anchorsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::ReorgYolo>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ReorgYolo>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReorgYolo node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto strides = origNode->get_strides();

    VPUX_THROW_UNLESS(strides.size() <= 2, "nGraph ReorgYolo node '{0}' has unsupported number of strides '{1}'",
                      origNode->get_friendly_name(), strides.size());
    VPUX_THROW_UNLESS(strides.front() == strides.back(),
                      "nGraph ReorgYolo node '{0}' has different strides '{1}' != '{2}'", origNode->get_friendly_name(),
                      strides.front(), strides.back());

    const auto strideAttr = getIntAttr(_ctx, strides.front());

    auto op = builder.create<IE::ReorgYoloOp>(createLocation(origNode), inputs[0], strideAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::DetectionOutput>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 5,
                      "nGraph DetectionOutput node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto detectionOutputAttr = importDetectionOutputAttrs(origNode->get_attrs());

    IE::DetectionOutputOp op;
    if (inputs.size() == 3) {
        op = builder.create<IE::DetectionOutputOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                                   nullptr, detectionOutputAttr);
    } else {
        op = builder.create<IE::DetectionOutputOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                                   inputs[4], detectionOutputAttr);
    }
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::NormalizeL2>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::NormalizeL2>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph NormalizeL2 node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());
    const auto epsModeAttr = importEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::NormalizeL2Op>(createLocation(origNode), inputs[0], inputs[1], nullptr, epsAttr,
                                                epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::CumSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::CumSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1 || inputs.size() == 2,
                      "nGraph CumSum node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto exclusiveAttr = origNode->is_exclusive() ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto reverseAttr = origNode->is_reverse() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::CumSumOp>(createLocation(origNode), inputs[0],
                                           (inputs.size() == 2) ? inputs[1] : nullptr, nullptr, exclusiveAttr,
                                           reverseAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::Eye>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 4,
                      "nGraph Eye node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto outputTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_out_type()));

    auto op = builder.create<IE::EyeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                        (inputs.size() == 4) ? inputs[3] : nullptr, nullptr, nullptr, nullptr,
                                        outputTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::MVN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph MVN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto normalize_varianceAttr = mlir::BoolAttr::get(_ctx, origNode->get_normalize_variance());
    const auto across_channelsAttr = mlir::BoolAttr::get(_ctx, origNode->get_across_channels());
    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());

    auto op = builder.create<IE::MVNOp>(createLocation(origNode), inputs[0], across_channelsAttr,
                                        normalize_varianceAttr, epsAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::MVN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph MVN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto normalize_varianceAttr = mlir::BoolAttr::get(_ctx, origNode->get_normalize_variance());

    const auto epsAttr = getFPAttr(_ctx, origNode->get_eps());
    const auto epsModeAttr = importMvnEpsMode(origNode->get_eps_mode());

    auto op = builder.create<IE::MVN6Op>(createLocation(origNode), inputs[0], nullptr /*scale*/, nullptr /*bias*/,
                                         inputs[1], nullptr, normalize_varianceAttr, epsAttr, epsModeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PRelu>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::PRelu>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PRelu node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::PReluOp>(createLocation(origNode), inputs[0], inputs[1]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Swish>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Swish>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    vpux::IE::SwishOp op;

    if (inputs.size() == 1) {
        op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], nullptr, nullptr);
    } else if (inputs.size() == 2) {
        op = builder.create<IE::SwishOp>(createLocation(origNode), inputs[0], inputs[1], nullptr);
    } else {
        VPUX_THROW("nGraph Swish node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                   inputs.size());
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GRN>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::GRN>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph GRN node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto biasAttr = getFPAttr(_ctx, origNode->get_bias());

    auto op = builder.create<IE::GRNOp>(createLocation(origNode), inputs[0], biasAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Negative>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Negative>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Negative node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::NegativeOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SignOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::CTCGreedyDecoder>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::CTCGreedyDecoder>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph CTCGreedyDecoder node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderOp>(createLocation(origNode), inputs[0], inputs[1],
                                                     origNode->get_ctc_merge_repeated());
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset6::CTCGreedyDecoderSeqLen>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::CTCGreedyDecoderSeqLen>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    if (inputs.size() == 3) {
        auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1],
                                                               inputs[2], origNode->get_merge_repeated());
        addOutputs(origNode, op);
    } else if (inputs.size() == 2) {
        auto op = builder.create<IE::CTCGreedyDecoderSeqLenOp>(createLocation(origNode), inputs[0], inputs[1],
                                                               /*blankIndex*/ nullptr, origNode->get_merge_repeated());
        addOutputs(origNode, op);
    } else {
        VPUX_THROW("nGraph CTCGreedyDecoderSeqLen node '{0}' has unsupported number of inputs '{1}'",
                   origNode->get_friendly_name(), inputs.size());
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v1::Pad>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Pad>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    if (inputs.size() == 4) {
        auto op =
                builder.create<IE::PadOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3], nullptr,
                                          nullptr, nullptr, importPadMode(origNode->get_pad_mode()), nullptr, nullptr);
        addOutputs(origNode, op);
    } else if (inputs.size() == 3) {
        auto op =
                builder.create<IE::PadOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr, nullptr,
                                          nullptr, nullptr, importPadMode(origNode->get_pad_mode()), nullptr, nullptr);
        addOutputs(origNode, op);
    } else {
        VPUX_THROW("nGraph Pad node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                   inputs.size());
    }
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LSTMCell>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::LSTMCell>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    // 7 inputs: X, initial_hidden_state, initial_cell_state, W, R, B, P
    VPUX_THROW_UNLESS(inputs.size() == 7, "nGraph LSTMCell node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());

    // create the IE::LSTMCellOp with 6 inputs, ignoring the 7th input
    auto op = builder.create<IE::LSTMCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                             inputs[4], inputs[5], hiddenSizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::LSTMCell>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::LSTMCell>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 6, "nGraph LSTMCell node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());

    auto op = builder.create<IE::LSTMCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                             inputs[4], inputs[5], hiddenSizeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::LSTMSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset5::LSTMSequence>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 7, "nGraph LSTMSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph LSTMSequence node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh", "tanh"}),
                      "nGraph LSTMSequence node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto directionAttr = importRNNSequenceDirection(origNode->get_direction());

    const auto isDynamic = [](auto node) {
        return node.get_partial_shape().is_dynamic();
    };

    const auto hasDynamicInputs = llvm::any_of(origNode->inputs(), isDynamic);
    const auto hasDynamicOutputs = llvm::any_of(origNode->outputs(), isDynamic);

    mlir::IntegerAttr seqLenAttr = nullptr;
    if (!hasDynamicInputs && !hasDynamicOutputs) {
        auto seqLenNode = dynamic_cast<ov::opset7::Constant*>(origNode->input_value(3).get_node());
        VPUX_THROW_UNLESS(
                seqLenNode != nullptr,
                "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must be a Constant node",
                origNode->get_friendly_name());
        auto seqLenValues = seqLenNode->cast_vector<uint32_t>();
        VPUX_THROW_UNLESS(
                seqLenValues.size() > 0,
                "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must contain more than "
                "0 elements",
                origNode->get_friendly_name());
        const auto isAllLensSame =
                std::all_of(seqLenValues.cbegin(), seqLenValues.cend(), [&seqLenValues](const auto item) {
                    return seqLenValues[0] == item;
                });
        VPUX_THROW_UNLESS(isAllLensSame,
                          "nGraph LSTMSequence node '{0}' has unsupported sequenceLengths input. It must contain all "
                          "the same values",
                          origNode->get_friendly_name());
        seqLenAttr = getIntAttr(_ctx, checked_cast<uint32_t>(seqLenValues[0]));
    }

    auto op = builder.create<IE::LSTMSequenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[4],
                                                 inputs[5], inputs[6], seqLenAttr, directionAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Subtract>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Subtract>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SubtractOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type), nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalAnd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::LogicalAnd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::AndOp>(createLocation(origNode), inputs[0], inputs[1],
                                        importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Ceiling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Ceiling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Ceiling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CeilingOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Equal>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Equal>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::EqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                          importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Select>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Select>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph Select node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::SelectOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                           importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset9::NonMaxSuppression>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 5) || (inputs.size() == 6),
                      "nGraph NonMaxSuppression node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto softNmsSigma = inputs.size() == 6 ? inputs[5] : nullptr;
    int center_point_box = 0;
    switch (origNode->get_box_encoding()) {
    case ov::opset9::NonMaxSuppression::BoxEncodingType::CENTER:
        center_point_box = 1;
        break;
    case ov::opset9::NonMaxSuppression::BoxEncodingType::CORNER:
        center_point_box = 0;
        break;
    default:
        VPUX_THROW("NonMaxSuppression layer has unsupported box encoding");
    }
    const auto boxEncodingAttr = importBoxEncodingType(center_point_box);
    const auto sortResultDescendingAttr = origNode->get_sort_result_descending() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::NonMaxSuppressionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], softNmsSigma, boxEncodingAttr,
                                                      sortResultDescendingAttr, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode) {
    static_assert(
            std::is_same<std::decay<decltype(*origNode)>::type, ov::op::internal::NonMaxSuppressionIEInternal>::value,
            "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 5) || (inputs.size() == 6),
                      "nGraph NonMaxSuppressionIEInternal node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    const auto softNmsSigma = inputs.size() == 6 ? inputs[5] : nullptr;
    const auto boxEncodingAttr = importBoxEncodingType(origNode->m_center_point_box);
    const auto sortResultDescendingAttr = origNode->m_sort_result_descending ? mlir::UnitAttr::get(_ctx) : nullptr;
    auto op = builder.create<IE::NonMaxSuppressionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], softNmsSigma, boxEncodingAttr,
                                                      sortResultDescendingAttr, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RMS>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::internal::RMS>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    const auto epsilon = getFPAttr(_ctx, origNode->get_epsilon());
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph RMSNorm node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RMSOp>(createLocation(origNode), inputs[0], inputs[1], epsilon);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::internal::RoPE>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::internal::RoPE>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph RoPE node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::RoPEOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset14::ScaledDotProductAttention>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset14::ScaledDotProductAttention>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    const auto size = inputs.size();
    VPUX_THROW_UNLESS(size >= 3 && size <= 5,
                      "nGraph ScaledDotProductAttention node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    mlir::Value attentionMask = nullptr;
    mlir::Value scale = nullptr;

    if (size == 4) {
        attentionMask = inputs[3];
    } else if (size == 5) {  // NOLINT magic number
        // OpenVINO SDPA layer has no constructor for no-mask yes-scale case
        // Instead, it passes a scalar (0) to the attention mask input, indicating that the mask should not be added
        const auto maskType = mlir::cast<vpux::NDTypeInterface>(inputs[3].getType());
        if (maskType.getShape().totalSize() > 1) {
            attentionMask = inputs[3];
        }
        scale = inputs[4];
    }

    auto causal = origNode->get_causal() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::SDPAOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], attentionMask,
                                         scale, causal);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ReverseSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::ReverseSequence>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph ReverseSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto seqAttr = getIntAttr(_ctx, origNode->get_sequence_axis());
    const auto batchAttr = getIntAttr(_ctx, origNode->get_batch_axis());

    auto op = builder.create<IE::ReverseSequenceOp>(createLocation(origNode), inputs[0], inputs[1], seqAttr, batchAttr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::SoftPlus>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::SoftPlus>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SoftPlus node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SoftPlusOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Less>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Less>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Less node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LessOp>(createLocation(origNode), inputs[0], inputs[1],
                                         importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LessEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LessEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LessEqual node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LessEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                              importBroadcastType(autob.m_type));

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Greater>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::Greater>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph Greater node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::GreaterOp>(createLocation(origNode), inputs[0], inputs[1],
                                            importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::GreaterEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::GreaterEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph GreaterEqual node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::GreaterEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                                 importBroadcastType(autob.m_type));

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::NotEqual>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::NotEqual>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::NotEqualOp>(createLocation(origNode), inputs[0], inputs[1],
                                             importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DepthToSpace>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::DepthToSpace>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto blockSizeAttr = getIntAttr(_ctx, origNode->get_block_size());

    auto op = builder.create<IE::DepthToSpaceOp>(createLocation(origNode), inputs[0], blockSizeAttr,
                                                 importDepthToSpaceMode(origNode->get_mode()));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseAnd>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v13::BitwiseAnd>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph BitwiseAnd node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::BitwiseAndOp>(createLocation(origNode), inputs[0], inputs[1],
                                               importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v13::BitwiseOr>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph BitwiseOr node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::BitwiseOrOp>(createLocation(origNode), inputs[0], inputs[1],
                                              importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseXor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v13::BitwiseXor>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph BitwiseXor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::BitwiseXorOp>(createLocation(origNode), inputs[0], inputs[1],
                                               importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseNot>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v13::BitwiseNot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph BitwiseNot node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto inputType = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType());
    mlir::Operation* op;
    if (inputType.getElementType().isSignlessInteger()) {
        op = builder.create<IE::LogicalNotOp>(createLocation(origNode), inputs[0]);
    } else {
        op = builder.create<IE::BitwiseNotOp>(createLocation(origNode), inputs[0]);
    }

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalNot>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalNot>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::LogicalNotOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalOr>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalOr>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LogicalOr node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LogicalOrOp>(createLocation(origNode), inputs[0], inputs[1],
                                              importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::LogicalXor>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::LogicalXor>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph LogicalXor node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto& autob = origNode->get_autob();

    auto op = builder.create<IE::LogicalXorOp>(createLocation(origNode), inputs[0], inputs[1],
                                               importBroadcastType(autob.m_type));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::SpaceToDepth>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::SpaceToDepth>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph SpaceToDepth node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto blockSizeAttr = getIntAttr(_ctx, origNode->get_block_size());
    const auto modeAttr = importSpaceToDepthMode(origNode->get_mode());

    auto op = builder.create<IE::SpaceToDepthOp>(createLocation(origNode), inputs[0], blockSizeAttr, modeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::SpaceToBatch>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset7::SpaceToBatch>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 4, "nGraph SpaceToBatch node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SpaceToBatch>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                               nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::ExtractImagePatches>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ExtractImagePatches>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1,
                      "nGraph ExtractImagePatches node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto sizesAttr = getIntArrayAttr(_ctx, origNode->get_sizes());
    const auto stridesAttr = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto ratesAttr = getIntArrayAttr(_ctx, origNode->get_rates());
    const auto autoPadAttr = importPadType(origNode->get_auto_pad());

    auto op = builder.create<IE::ExtractImagePatchesOp>(createLocation(origNode), inputs[0], sizesAttr, stridesAttr,
                                                        ratesAttr, autoPadAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Abs>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Abs>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Abs node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AbsOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::HSigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::HSigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph HSigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HSigmoidOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Atan>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Atan>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Atan node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AtanOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Asin>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Asin>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Asin node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AsinOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Acos>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Acos>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Acos node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::AcosOp>(createLocation(origNode), inputs[0]);
    addOutputs(origNode, op);
}
void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::PSROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::PSROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph PSROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputDim = getIntAttr(_ctx, origNode->get_output_dim());
    const auto spatialScale = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto groupSize = getIntAttr(_ctx, origNode->get_group_size());
    const auto spatialBinsX = getIntAttr(_ctx, origNode->get_spatial_bins_x());
    const auto spatialBinsY = getIntAttr(_ctx, origNode->get_spatial_bins_y());
    const auto mode = importPSROIPoolingMode(origNode->get_mode());

    auto op = builder.create<IE::PSROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1], outputDim,
                                                 spatialScale, groupSize, spatialBinsX, spatialBinsY, mode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::HardSigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::HardSigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph HardSigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::HardSigmoidOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], nullptr,
                                                nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingBagOffsetsSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingBagOffsetsSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    IE::EmbeddingBagOffsetsSumOp op;
    mlir::Value defaultIndex = (inputs.size() >= 4) ? inputs[3] : nullptr;
    mlir::Value weights = (inputs.size() >= 5) ? inputs[4] : nullptr;
    op = builder.create<IE::EmbeddingBagOffsetsSumOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                      defaultIndex, weights, nullptr, nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingSegmentsSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingSegmentsSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4 || inputs.size() == 5 || inputs.size() == 6,
                      "nGraph EmbeddingSegmentsSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    mlir::Value defaultIndex = (inputs.size() >= 5) ? inputs[4] : nullptr;
    mlir::Value weights = (inputs.size() >= 6) ? inputs[5] : nullptr;
    auto op = builder.create<IE::EmbeddingSegmentsSumOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                         inputs[3], defaultIndex, weights, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr);

    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset3::EmbeddingBagPackedSum>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::EmbeddingBagPackedSum>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph EmbeddingBagPackedSum node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    mlir::Value weights = (inputs.size() == 3) ? inputs[2] : nullptr;
    auto op = builder.create<IE::EmbeddingBagPackedSumOp>(createLocation(origNode), inputs[0], inputs[1], weights);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::Assign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Assign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::AssignOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::Assign>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Assign node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::AssignOp>(createLocation(origNode), inputs[0], nameAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::ReadValue>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph ReadValue node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());

    auto op = builder.create<IE::ReadValueOp>(createLocation(origNode), inputs[0], nameAttr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v6::ReadValue>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() <= 1, "nGraph ReadValue node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto nameAttr = mlir::StringAttr::get(_ctx, origNode->get_variable_id());
    const auto elementType = importPrecision(_ctx, origNode->get_element_type());
    const auto shape = importShape(origNode->get_shape());
    mlir::Value inputTensor;
    if (inputs.size() == 0) {
        // If the 1st input is not provided, ReadValue returns the tensor with zero values
        const auto inputType = mlir::RankedTensorType::get({shape}, elementType);
        inputTensor = Const::createZerosConst(builder, createLocation(origNode), inputType);
    } else {
        inputTensor = inputs[0];
    }

    auto op = builder.create<IE::ReadValueOp>(createLocation(origNode), inputTensor, nameAttr,
                                              mlir::TypeAttr::get(elementType), getIntArrayAttr(_ctx, shape));
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::GRUCell>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::GRUCell>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 4 || inputs.size() == 5,
                      "nGraph GRUCell node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph GRUCell node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh"}),
                      "nGraph GRUCell node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());
    const auto clipAttr = getFPAttr(_ctx, origNode->get_clip());
    const auto shouldLinearBeforeResetAttr = origNode->get_linear_before_reset() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto biasesInput = inputs.size() == 5 ? inputs[4] : nullptr;
    auto op = builder.create<IE::GRUCellOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], inputs[3],
                                            biasesInput, hiddenSizeAttr, shouldLinearBeforeResetAttr, clipAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::GRUSequence>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v5::GRUSequence>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 6, "nGraph GRUSequence node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    VPUX_THROW_UNLESS(origNode->get_clip() == 0.0f, "nGraph GRUSequence node '{0}' has unsupported clip value '{1}'",
                      origNode->get_friendly_name(), origNode->get_clip());

    VPUX_THROW_UNLESS(origNode->get_activations() == std::vector<std::string>({"sigmoid", "tanh"}),
                      "nGraph GRUSequence node '{0}' has unsupported activations '{1}'", origNode->get_friendly_name(),
                      origNode->get_activations());

    const auto seqLenConstant = dynamic_cast<ov::opset7::Constant*>(origNode->input_value(2).get_node());
    VPUX_THROW_UNLESS(seqLenConstant != nullptr,
                      "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must be a Constant node",
                      origNode->get_friendly_name());
    const auto seqLenValues = seqLenConstant->cast_vector<uint32_t>();
    VPUX_THROW_UNLESS(seqLenValues.size() > 0,
                      "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must contain more than "
                      "0 elements",
                      origNode->get_friendly_name());
    const auto isAllLensSame =
            std::all_of(seqLenValues.cbegin(), seqLenValues.cend(), [&seqLenValues](const auto item) {
                return seqLenValues[0] == item;
            });
    VPUX_THROW_UNLESS(
            isAllLensSame,
            "nGraph GRUSequence node '{0}' has unsupported sequenceLengths input. It must contain all the same values",
            origNode->get_friendly_name());
    const auto seqLenAttr = getIntAttr(_ctx, checked_cast<uint32_t>(seqLenValues[0]));

    VPUX_THROW_UNLESS(origNode->get_direction() == ov::op::RecurrentSequenceDirection::FORWARD ||
                              origNode->get_direction() == ov::op::RecurrentSequenceDirection::REVERSE,
                      "nGraph GRUSequence node '{0}' supports direction 'FORWARD' and 'REVERSE'",
                      origNode->get_friendly_name());
    const auto directionAttr = importRNNSequenceDirection(origNode->get_direction());

    const auto hiddenSizeAttr = getIntAttr(_ctx, origNode->get_hidden_size());
    const auto clipAttr = getFPAttr(_ctx, origNode->get_clip());
    const auto shouldLinearBeforeResetAttr = origNode->get_linear_before_reset() ? mlir::UnitAttr::get(_ctx) : nullptr;

    auto op = builder.create<IE::GRUSequenceOp>(createLocation(origNode), inputs[0], inputs[1], inputs[3], inputs[4],
                                                inputs[5], hiddenSizeAttr, seqLenAttr, directionAttr,
                                                shouldLinearBeforeResetAttr, clipAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset1::DeformablePSROIPooling>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v1::DeformablePSROIPooling>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "nGraph DeformablePSROIPooling node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outputDim = getIntAttr(_ctx, origNode->get_output_dim());
    const auto spatialScale = getFPAttr(_ctx, origNode->get_spatial_scale());
    const auto groupSize = getIntAttr(_ctx, origNode->get_group_size());
    const auto spatialBinsX = getIntAttr(_ctx, origNode->get_spatial_bins_x());
    const auto spatialBinsY = getIntAttr(_ctx, origNode->get_spatial_bins_y());
    const auto transStd = getFPAttr(_ctx, origNode->get_trans_std());
    const auto partSize = getIntAttr(_ctx, origNode->get_part_size());
    const auto mode = importDeformablePSROIPoolingMode(origNode->get_mode());

    mlir::Value inputTransformations = (inputs.size() == 3) ? inputs[2] : nullptr;
    auto op = builder.create<IE::DeformablePSROIPoolingOp>(createLocation(origNode), inputs[0], inputs[1],
                                                           inputTransformations, outputDim, spatialScale, groupSize,
                                                           spatialBinsX, spatialBinsY, transStd, partSize, mode);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::DFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::DFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                        (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::RDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::RDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                         (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::IDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v7::IDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::IDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                         (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v9::IRDFT>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(((inputs.size() == 2) || (inputs.size() == 3)),
                      "nGraph Interpolate node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto op = builder.create<IE::IRDFTOp>(createLocation(origNode), inputs[0], inputs[1],
                                          (inputs.size() == 3) ? inputs[2] : nullptr, nullptr, nullptr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::Range>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v4::Range>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());
    auto outputShape = origNode->get_output_partial_shape(0);

    // Output shape is static when inputs are all constant.
    if (outputShape.is_static()) {
        auto outputType = importTensor(origNode->get_output_partial_shape(0), origNode->get_output_element_type(0));
        auto constStart = getParentNodeAs<ov::op::v0::Constant>(origNode->input(0));
        auto constStop = getParentNodeAs<ov::op::v0::Constant>(origNode->input(1));
        auto constStep = getParentNodeAs<ov::op::v0::Constant>(origNode->input(2));
        auto start = constStart->cast_vector<float>();
        auto stop = constStop->cast_vector<float>();
        auto step = constStep->cast_vector<float>();
        auto outPrecision = origNode->get_output_element_type(0);
        auto rangeOutputAttr = calRangeOutputAttr(start[0], stop[0], step[0], outputType, outPrecision);
        auto constOp = builder.create<Const::DeclareOp>(createLocation(origNode), outputType,
                                                        Const::ContentAttr::get(rangeOutputAttr));
        addOutputs(origNode, constOp);
    }
    const auto outElemTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_output_type()));
    auto op = builder.create<IE::RangeOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2], outElemTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::NonZero>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v3::NonZero>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outElemTypeAttr = mlir::TypeAttr::get(importPrecision(_ctx, origNode->get_output_type()));
    auto op = builder.create<IE::NonZeroOp>(createLocation(origNode), inputs[0], outElemTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset8::If>::value,
                  "opset operation mismatch");

    auto* ctx = builder.getContext();
    const auto inputs = getInputs(origNode);
    const auto location = createLocation(origNode);

    const auto& origNodeThenBody = origNode->get_then_body();
    const auto& origNodeElseBody = origNode->get_else_body();
    NGraphImporter importerThen(ctx, origNodeThenBody, false, _log);
    NGraphImporter importerElse(ctx, origNodeElseBody, false, _log);

    SmallVector<mlir::Type> results;
    results.append(importerThen.getRegionResults());

    auto op = builder.create<IE::IfOp>(location, results, inputs);
    addOutputs(origNode, op);

    auto thenBlock = &op.getThenBranch().emplaceBlock();
    auto thenBranchBuilder = mlir::OpBuilder::atBlockBegin(&op.getThenBranch().front(), nullptr);
    importerThen.buildBlockFromRegion(appendLoc(location, "then"), thenBranchBuilder, thenBlock);

    auto elseBlock = &op.getElseBranch().emplaceBlock();
    auto elseBranchBuilder = mlir::OpBuilder::atBlockBegin(&op.getElseBranch().front(), nullptr);
    importerElse.buildBlockFromRegion(appendLoc(location, "else"), elseBranchBuilder, elseBlock);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShapeOf>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset1::ShapeOf>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto outElemTypeAttr =
            mlir::TypeAttr::get(importPrecision(builder.getContext(), origNode->output(0).get_element_type()));
    auto op = builder.create<IE::ShapeOfOp>(createLocation(origNode), inputs[0], outElemTypeAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TensorIterator>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset1::TensorIterator>::value,
                  "opset operation mismatch");

    auto* ctx = builder.getContext();
    const auto inputs = getInputs(origNode);
    const auto location = createLocation(origNode);

    const auto& origNodeBodyModule = origNode->get_body();
    NGraphImporter importerBodyModule(ctx, origNodeBodyModule, false, _log);

    const auto numIterationsAttr = getIntAttr(ctx, origNode->get_num_iterations());
    SmallVector<mlir::Attribute> sliceInputVector, invariantInputVector, mergedInputVector, concatOutputVector,
            invariantOutputVector;
    for (auto& inDesc : origNode->get_input_descriptions()) {
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset1::TensorIterator::SliceInputDescription>(inDesc)) {
            auto input = inputs[desc->m_input_index];
            sliceInputVector.push_back(importSliceInputPortMapAttr(ctx, desc, input, origNode->get_num_iterations()));
            continue;
        }
        if (const auto& desc =
                    std::dynamic_pointer_cast<ov::opset1::TensorIterator::InvariantInputDescription>(inDesc)) {
            invariantInputVector.push_back(importInvariantInputPortMapAttr(ctx, desc));
            continue;
        }
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset1::TensorIterator::MergedInputDescription>(inDesc)) {
            mergedInputVector.push_back(importMergedInputPortMapAttr(ctx, desc));
            continue;
        }
        VPUX_THROW("Unsupported input description for tensor iterator op input at index {0}", inDesc->m_input_index);
    }

    for (auto& outDesc : origNode->get_output_descriptions()) {
        if (const auto& desc =
                    std::dynamic_pointer_cast<ov::opset1::TensorIterator::ConcatOutputDescription>(outDesc)) {
            concatOutputVector.push_back(importConcatOutputPortMapAttr(ctx, desc));
            continue;
        }
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset1::TensorIterator::BodyOutputDescription>(outDesc)) {
            invariantOutputVector.push_back(importBodyOutputPortMapAttr(ctx, desc));
            continue;
        }
        VPUX_THROW("Unsupported output description for tensor iterator op output at index {0}",
                   outDesc->m_output_index);
    }

    SmallVector<mlir::Type> results;
    results.append(importerBodyModule.getLoopLikeRegionResults(
            origNode->get_num_iterations(), origNode->get_output_size(), concatOutputVector, invariantOutputVector));

    VPUX_THROW_UNLESS(inputs.size() == sliceInputVector.size() + invariantInputVector.size() + mergedInputVector.size(),
                      "nGraph TensorIterator node '{0}' has incompatible input number {1} with its body inputs "
                      "descriptions sliceInput '{2}' + invariantInput '{3}' + mergedInput '{4}'",
                      origNode->get_friendly_name(), inputs.size(), sliceInputVector.size(),
                      invariantInputVector.size(), mergedInputVector.size());

    VPUX_THROW_UNLESS(results.size() == concatOutputVector.size() + invariantOutputVector.size(),
                      "nGraph TensorIterator node '{0}' has incompatible output number {1} with its body outputs "
                      "descriptions concatOutput '{2}' + invariantOutput '{3}'",
                      origNode->get_friendly_name(), results.size(), concatOutputVector.size(),
                      invariantOutputVector.size());

    const auto sliceInputDescs = mlir::ArrayAttr::get(ctx, sliceInputVector);
    const auto invariantInputDescs = mlir::ArrayAttr::get(ctx, invariantInputVector);
    const auto feedbackInputDescs = mlir::ArrayAttr::get(ctx, mergedInputVector);
    const auto concatOutputDescs = mlir::ArrayAttr::get(ctx, concatOutputVector);
    const auto invariantOutputDescs = mlir::ArrayAttr::get(ctx, invariantOutputVector);

    auto op = builder.create<IE::TensorIteratorOp>(location, results, inputs, numIterationsAttr, sliceInputDescs,
                                                   invariantInputDescs, feedbackInputDescs, concatOutputDescs,
                                                   invariantOutputDescs);
    addOutputs(origNode, op);

    auto bodyModuleBlock = &op.getBodyModule().emplaceBlock();
    auto bodyModuleBuilder = mlir::OpBuilder::atBlockBegin(&op.getBodyModule().front(), nullptr);
    importerBodyModule.buildBlockFromBody(appendLoc(location, "body"), bodyModuleBuilder, bodyModuleBlock);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset5::Loop>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset5::Loop>::value,
                  "opset operation mismatch");

    auto* ctx = builder.getContext();
    const auto inputs = getInputs(origNode);
    const auto location = createLocation(origNode);

    const auto& origNodeBodyModule = origNode->get_function();
    NGraphImporter importerBodyModule(ctx, origNodeBodyModule, false, _log);

    const auto numIterationsAttr = getIntAttr(ctx, origNode->get_num_iterations());
    const auto currentIterIndexInt = origNode->get_special_body_ports().current_iteration_input_idx;
    const auto currentIterIndexAttr = getIntAttr(ctx, currentIterIndexInt);

    const auto execCondIndexInt = origNode->get_special_body_ports().body_condition_output_idx;
    VPUX_THROW_WHEN(execCondIndexInt == -1, "exec_cond_index cannot be -1!");
    const auto execCondIndexAttr = getIntAttr(ctx, execCondIndexInt);

    SmallVector<mlir::Attribute> sliceInputVector, invariantInputVector, mergedInputVector, concatOutputVector,
            invariantOutputVector;
    for (auto& inDesc : origNode->get_input_descriptions()) {
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset5::Loop::SliceInputDescription>(inDesc)) {
            auto input = inputs[desc->m_input_index];
            sliceInputVector.push_back(importSliceInputPortMapAttr(ctx, desc, input, origNode->get_num_iterations()));
            continue;
        }
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset5::Loop::InvariantInputDescription>(inDesc)) {
            invariantInputVector.push_back(importInvariantInputPortMapAttr(ctx, desc));
            continue;
        }
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset5::Loop::MergedInputDescription>(inDesc)) {
            mergedInputVector.push_back(importMergedInputPortMapAttr(ctx, desc));
            continue;
        }
        VPUX_THROW("Unsupported input description for Loop op input at index {0}", inDesc->m_input_index);
    }

    for (auto& outDesc : origNode->get_output_descriptions()) {
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset5::Loop::ConcatOutputDescription>(outDesc)) {
            concatOutputVector.push_back(importConcatOutputPortMapAttr(ctx, desc));
            continue;
        }
        if (const auto& desc = std::dynamic_pointer_cast<ov::opset5::Loop::BodyOutputDescription>(outDesc)) {
            invariantOutputVector.push_back(importBodyOutputPortMapAttr(ctx, desc));
            continue;
        }
        VPUX_THROW("Unsupported output description for Loop op output at index {0}", outDesc->m_output_index);
    }

    SmallVector<mlir::Type> results;
    results.append(importerBodyModule.getLoopLikeRegionResults(
            origNode->get_num_iterations(), origNode->get_output_size(), concatOutputVector, invariantOutputVector));

    VPUX_THROW_UNLESS(
            inputs.size() - 2 == sliceInputVector.size() + invariantInputVector.size() + mergedInputVector.size(),
            "nGraph Loop node '{0}' has incompatible input number {1} with its body inputs "
            "descriptions sliceInput '{2}' + invariantInput '{3}' + mergedInput '{4}'",
            origNode->get_friendly_name(), inputs.size(), sliceInputVector.size(), invariantInputVector.size(),
            mergedInputVector.size());

    VPUX_THROW_UNLESS(results.size() == concatOutputVector.size() + invariantOutputVector.size(),
                      "nGraph Loop node '{0}' has incompatible output number {1} with its body outputs "
                      "descriptions concatOutput '{2}' + invariantOutput '{3}'",
                      origNode->get_friendly_name(), results.size(), concatOutputVector.size(),
                      invariantOutputVector.size());

    const auto sliceInputDescs = mlir::ArrayAttr::get(ctx, sliceInputVector);
    const auto invariantInputDescs = mlir::ArrayAttr::get(ctx, invariantInputVector);
    const auto feedbackInputDescs = mlir::ArrayAttr::get(ctx, mergedInputVector);
    const auto concatOutputDescs = mlir::ArrayAttr::get(ctx, concatOutputVector);
    const auto invariantOutputDescs = mlir::ArrayAttr::get(ctx, invariantOutputVector);

    auto op = builder.create<IE::LoopOp>(location, results, inputs, numIterationsAttr, currentIterIndexAttr,
                                         execCondIndexAttr, sliceInputDescs, invariantInputDescs, feedbackInputDescs,
                                         concatOutputDescs, invariantOutputDescs);
    addOutputs(origNode, op);

    auto bodyModuleBlock = &op.getBodyModule().emplaceBlock();
    auto bodyModuleBuilder = mlir::OpBuilder::atBlockBegin(&op.getBodyModule().front(), nullptr);
    importerBodyModule.buildBlockFromBody(appendLoc(location, "body"), bodyModuleBuilder, bodyModuleBlock);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset14::Inverse>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset14::Inverse>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Inverse node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    const auto adjointAttr = origNode->get_adjoint() ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto op = builder.create<IE::InverseOp>(createLocation(origNode), inputs[0], adjointAttr);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset8::DeformableConvolution>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v8::DeformableConvolution>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS((inputs.size() == 3 || inputs.size() == 4),
                      "nGraph node '{0}' has unsupported number of inputs '{1}'", origNode->get_friendly_name(),
                      inputs.size());

    const auto attrStride = getIntArrayAttr(_ctx, origNode->get_strides());
    const auto attrPadsBegin = getIntArrayAttr(_ctx, origNode->get_pads_begin());
    const auto attrPadsEnd = getIntArrayAttr(_ctx, origNode->get_pads_end());
    const auto attrDilation = getIntArrayAttr(_ctx, origNode->get_dilations());
    const auto attrBilinearInterpolationPad =
            origNode->get_bilinear_interpolation_pad() ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto attrGroup = origNode->get_group() ? getIntAttr(_ctx, origNode->get_group()) : nullptr;
    const auto attrDeformableGroup =
            origNode->get_deformable_group() ? getIntAttr(_ctx, origNode->get_deformable_group()) : nullptr;
    auto attrGroupValue = (attrGroup == nullptr) ? builder.getIntegerAttr(builder.getIntegerType(32), 1) : attrGroup;
    auto deformableGroupValue = (attrDeformableGroup == nullptr) ? builder.getIntegerAttr(builder.getIntegerType(32), 1)
                                                                 : attrDeformableGroup;
    auto op = builder.create<IE::DeformableConvolutionOp>(createLocation(origNode), inputs[0], inputs[1], inputs[2],
                                                          (inputs.size() == 4) ? inputs[3] : nullptr, attrStride,
                                                          attrPadsBegin, attrPadsEnd, attrDilation, attrGroupValue,
                                                          deformableGroupValue, attrBilinearInterpolationPad);
    addOutputs(origNode, op);
}

void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::VariadicSplit>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::opset1::VariadicSplit>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 3, "nGraph node '{0}' has unsupported number of inputs '{1}'.",
                      origNode->get_friendly_name(), inputs.size());

    // Assumed to be always constant
    const auto axisNode = ov::as_type_ptr<ov::op::v0::Constant>(origNode->input_value(1).get_node_shared_ptr());
    VPUX_THROW_UNLESS(axisNode != nullptr,
                      "nGraph VariadicSplit node '{0}' has unsupported axis input. It must be a Constant node.",
                      origNode->get_friendly_name());
    const auto axis = axisNode->cast_vector<int64_t>().front();

    const auto splitLengthsNode = ov::as_type_ptr<ov::op::v0::Constant>(origNode->input_value(2).get_node_shared_ptr());
    VPUX_THROW_UNLESS(
            splitLengthsNode != nullptr,
            "nGraph VariadicSplit node '{0}' has unsupported split_lengths input. It must be a Constant node.",
            origNode->get_friendly_name());
    const auto splitLengths = splitLengthsNode->cast_vector<int64_t>();

    const auto input = inputs[0];
    auto op = builder.create<IE::VariadicSplitOp>(createLocation(origNode), input, axis,
                                                  getIntArrayAttr(builder, splitLengths));

    addOutputs(origNode, op);
}

// Node parsing for default OV Op dispatch
// Such cases are treated as OV Custom Operators
void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::Op>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::Op>::value, "opset operation mismatch");

    const auto inputs = getInputs(origNode);

    CustomOperatorAttrVisitor collector(_ctx);
    origNode->visit_attributes(collector);

    // Use Node type as the UniqueId for the CustomOp
    auto customOpName = origNode->get_type_name();
    auto opUniqueIdStrAttr = mlir::StringAttr::get(_ctx, customOpName);

    auto dictAttr = collector.getAttrDict();
    auto kernelPathAttr = collector.getKernelPath();
    VPUX_THROW_UNLESS(kernelPathAttr != nullptr,
                      "Node {0} doesn't respect Custom Operator format or is an unrecognized operator.");

    auto outputsSize = origNode->get_output_size();
    SmallVector<mlir::Type> outputTypes;
    for (auto i : irange(outputsSize)) {
        auto type = importTensor(origNode->get_output_shape(i), origNode->get_output_element_type(i));
        outputTypes.push_back(type);
    }

    auto op =
            builder.create<IE::ExternalKernelOp>(createLocation(origNode), mlir::TypeRange(outputTypes),
                                                 mlir::ValueRange(inputs), dictAttr, opUniqueIdStrAttr, kernelPathAttr);

    addOutputs(origNode, op);
}

//
// IR builder helpers
//

SmallVector<mlir::Value> NGraphImporter::getInputs(const OrigNodePtr& node) {
    SmallVector<mlir::Value> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        out.push_back(_importedVals.at(input.get_source_output()));
    }

    return out;
}

void NGraphImporter::addOutputs(const OrigNodePtr& node, mlir::Operation* op) {
    const auto results = op->getOpResults();

    VPUX_THROW_UNLESS(
            results.size() == node->get_output_size(),
            "Mismatch between original Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
            node->get_friendly_name(), node->get_output_size(), results.size());

    for (const auto& res : results) {
        _importedVals.emplace(node->output(res.getResultNumber()), res);
    }
}

mlir::Location NGraphImporter::createLocation(const OrigNodePtr& node) {
    return vpux::IE::createLayerLocation(_ctx, node->get_friendly_name(), node->get_type_name());
}

//
// nGraph attributes importers
//

mlir::RankedTensorType NGraphImporter::importTensor(const ov::PartialShape& shape, const ov::element::Type& elemType) {
    return mlir::RankedTensorType::get(ArrayRef(importShape(shape)), importPrecision(_ctx, elemType));
}

IE::AutoBroadcastTypeAttr NGraphImporter::importBroadcastType(ov::op::AutoBroadcastType bType) {
    switch (bType) {
    case ov::op::AutoBroadcastType::NONE:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    case ov::op::AutoBroadcastType::NUMPY:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::NUMPY);
    case ov::op::AutoBroadcastType::PDPD:
        return IE::AutoBroadcastTypeAttr::get(_ctx, IE::AutoBroadcastType::PDPD);
    default:
        VPUX_THROW("Unknown AutoBroadcastType");
    }
}

IE::BroadcastTypeAttr NGraphImporter::importBroadcastMode(ov::op::BroadcastType bType) {
    switch (bType) {
    case ov::op::BroadcastType::NUMPY:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::NUMPY);
    case ov::op::BroadcastType::EXPLICIT:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::EXPLICIT);
    case ov::op::BroadcastType::BIDIRECTIONAL:
        return IE::BroadcastTypeAttr::get(_ctx, IE::BroadcastType::BIDIRECTIONAL);
    default:
        VPUX_THROW("Unknown BroadcastMode");
    }
}

IE::RoundingTypeAttr NGraphImporter::importRoundingType(ov::op::RoundingType roundingType) {
    switch (roundingType) {
    case ov::op::RoundingType::FLOOR:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::FLOOR);
    case ov::op::RoundingType::CEIL:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    case ov::op::RoundingType::CEIL_TORCH:
        // `CEIL_TORCH` is not yet supported in OpenVINO.
        // In pass `ConvertMaxPool14ToMaxPool8`, OpenVino uses `CEIL` instead.
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

IE::EpsModeAttr NGraphImporter::importEpsMode(ov::op::EpsMode val) {
    switch (val) {
    case ov::op::EpsMode::ADD:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::ADD);
    case ov::op::EpsMode::MAX:
        return IE::EpsModeAttr::get(_ctx, IE::EpsMode::MAX);
    default:
        VPUX_THROW("Unknown EpsMode");
    }
}

IE::MvnEpsModeAttr NGraphImporter::importMvnEpsMode(ov::op::MVNEpsMode val) {
    switch (val) {
    case ov::op::MVNEpsMode::INSIDE_SQRT:
        return IE::MvnEpsModeAttr::get(_ctx, IE::MvnEpsMode::INSIDE_SQRT);
    case ov::op::MVNEpsMode::OUTSIDE_SQRT:
        return IE::MvnEpsModeAttr::get(_ctx, IE::MvnEpsMode::OUTSIDE_SQRT);
    default:
        VPUX_THROW("Unknown MvnEpsMode");
    }
}

IE::TopKModeAttr NGraphImporter::importTopKMode(ov::op::TopKMode val) {
    switch (val) {
    case ov::op::TopKMode::MAX:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MAX);
    case ov::op::TopKMode::MIN:
        return IE::TopKModeAttr::get(_ctx, IE::TopKMode::MIN);
    default:
        VPUX_THROW("Unknown TopKMode");
    }
}

IE::TopKSortTypeAttr NGraphImporter::importTopKSortType(ov::op::TopKSortType val) {
    switch (val) {
    case ov::op::TopKSortType::NONE:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::NONE);
    case ov::op::TopKSortType::SORT_INDICES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_INDICES);
    case ov::op::TopKSortType::SORT_VALUES:
        return IE::TopKSortTypeAttr::get(_ctx, IE::TopKSortType::SORT_VALUES);
    default:
        VPUX_THROW("Unknown TopKSortType");
    }
}

IE::GridSampleModeAttr NGraphImporter::importGridSampleMode(const ov::op::v9::GridSample::InterpolationMode& val) {
    IE::GridSampleModeAttr attr;
    if (val == ov::op::v9::GridSample::InterpolationMode::BILINEAR) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::BILINEAR);
    } else if (val == ov::op::v9::GridSample::InterpolationMode::BICUBIC) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::BICUBIC);
    } else if (val == ov::op::v9::GridSample::InterpolationMode::NEAREST) {
        attr = IE::GridSampleModeAttr::get(_ctx, IE::GridSampleMode::NEAREST);
    } else {
        VPUX_THROW("Unknown GridSampleMode");
    }
    return attr;
}

IE::GridSamplePaddingModeAttr NGraphImporter::importGridSamplePaddingMode(
        const ov::op::v9::GridSample::PaddingMode& val) {
    IE::GridSamplePaddingModeAttr attr;
    if (val == ov::op::v9::GridSample::PaddingMode::ZEROS) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::ZEROS);
    } else if (val == ov::op::v9::GridSample::PaddingMode::BORDER) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::BORDER);
    } else if (val == ov::op::v9::GridSample::PaddingMode::REFLECTION) {
        attr = IE::GridSamplePaddingModeAttr::get(_ctx, IE::GridSamplePaddingMode::REFLECTION);
    } else {
        VPUX_THROW("Unknown GridSamplePaddingMode");
    }
    return attr;
}

IE::ProposalAttr NGraphImporter::importProposalAttrs(const ov::op::v0::Proposal::Attributes& val) {
    const auto baseSizeAttr = getIntAttr(_ctx, val.base_size);
    const auto preNmsTopNAttr = getIntAttr(_ctx, val.pre_nms_topn);
    const auto postNmsTopNAttr = getIntAttr(_ctx, val.post_nms_topn);
    const auto nmsThreshNAttr = getFPAttr(_ctx, val.nms_thresh);
    const auto featStrideAttr = getIntAttr(_ctx, val.feat_stride);
    const auto minSizeNAttr = getIntAttr(_ctx, val.min_size);
    const auto ratioAttr = getFPArrayAttr(_ctx, val.ratio);
    const auto scaleAttr = getFPArrayAttr(_ctx, val.scale);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto normalizeAttr = mlir::BoolAttr::get(_ctx, val.normalize);
    const auto boxSizeScaleAttr = getFPAttr(_ctx, val.box_size_scale);
    const auto boxCoordinateScaleAttr = getFPAttr(_ctx, val.box_coordinate_scale);
    const auto frameworkAttr = mlir::StringAttr::get(_ctx, val.framework);
    const auto inferProbsAttr = mlir::BoolAttr::get(_ctx, val.infer_probs);

    return IE::ProposalAttr::get(_ctx, baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshNAttr, featStrideAttr,
                                 minSizeNAttr, ratioAttr, scaleAttr, clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                 boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr, inferProbsAttr);
}

IE::InterpolateAttr NGraphImporter::importInterpolateAttrs(const ov::opset4::Interpolate::InterpolateAttrs& val) {
    // mode
    IE::InterpolateModeAttr modeAttr;
    switch (val.mode) {
    case ov::opset4::Interpolate::InterpolateMode::NEAREST:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::NEAREST);
        break;
    case ov::opset4::Interpolate::InterpolateMode::LINEAR:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::LINEAR);
        break;
    case ov::opset4::Interpolate::InterpolateMode::LINEAR_ONNX:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::LINEAR_ONNX);
        break;
    case ov::opset4::Interpolate::InterpolateMode::CUBIC:
        modeAttr = IE::InterpolateModeAttr::get(_ctx, IE::InterpolateMode::CUBIC);
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    IE::InterpolateCalcModeAttr calcModeAttr;
    switch (val.shape_calculation_mode) {
    case ov::opset4::Interpolate::ShapeCalcMode::SIZES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::SIZES);
        break;
    case ov::opset4::Interpolate::ShapeCalcMode::SCALES:
        calcModeAttr = IE::InterpolateCalcModeAttr::get(_ctx, IE::InterpolateCalcMode::SCALES);
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    IE::InterpolateCoordModeAttr coordModeAttr;
    switch (val.coordinate_transformation_mode) {
    case ov::opset4::Interpolate::CoordinateTransformMode::HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::HALF_PIXEL);
        break;
    case ov::opset4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL);
        break;
    case ov::opset4::Interpolate::CoordinateTransformMode::ASYMMETRIC:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::ASYMMETRIC);
        break;
    case ov::opset4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN);
        break;
    case ov::opset4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS:
        coordModeAttr = IE::InterpolateCoordModeAttr::get(_ctx, IE::InterpolateCoordMode::ALIGN_CORNERS);
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    IE::InterpolateNearestModeAttr nearestModeAttr;
    switch (val.nearest_mode) {
    case ov::opset4::Interpolate::NearestMode::ROUND_PREFER_FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ROUND_PREFER_FLOOR);
        break;
    case ov::opset4::Interpolate::NearestMode::ROUND_PREFER_CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::ROUND_PREFER_CEIL);
        break;
    case ov::opset4::Interpolate::NearestMode::FLOOR:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::FLOOR);
        break;
    case ov::opset4::Interpolate::NearestMode::CEIL:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::CEIL);
        break;
    case ov::opset4::Interpolate::NearestMode::SIMPLE:
        nearestModeAttr = IE::InterpolateNearestModeAttr::get(_ctx, IE::InterpolateNearestMode::SIMPLE);
        break;
    default:
        VPUX_THROW("Unsupported interpolate nearest mode");
    }

    const auto antialiasAttr = mlir::BoolAttr::get(_ctx, val.antialias);
    const auto padsBeginAttr = getIntArrayAttr(_ctx, val.pads_begin);
    const auto padsEndAttr = getIntArrayAttr(_ctx, val.pads_end);
    const auto cubeCoeffAttr = getFPAttr(_ctx, val.cube_coeff);

    return IE::InterpolateAttr::get(_ctx, modeAttr, calcModeAttr, coordModeAttr, nearestModeAttr, antialiasAttr,
                                    padsBeginAttr, padsEndAttr, cubeCoeffAttr);
}

IE::ReverseModeAttr NGraphImporter::importReverseMode(const ov::op::v1::Reverse::Mode mode) {
    IE::ReverseModeAttr attr;
    if (mode == ov::op::v1::Reverse::Mode::INDEX) {
        attr = IE::ReverseModeAttr::get(_ctx, IE::ReverseMode::INDEX);
    } else if (mode == ov::op::v1::Reverse::Mode::MASK) {
        attr = IE::ReverseModeAttr::get(_ctx, IE::ReverseMode::MASK);
    } else {
        VPUX_THROW("Unknown ReverseModeAttr");
    }
    return attr;
}

IE::DetectionOutputCodeTypeAttr NGraphImporter::importDetectionOutputCodeType(const std::string& codeType) {
    if (codeType == "caffe.PriorBoxParameter.CENTER_SIZE") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CENTER_SIZE);
    } else if (codeType == "caffe.PriorBoxParameter.CORNER") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CORNER);
    } else if (codeType == "caffe.PriorBoxParameter.CORNER_SIZE") {
        return IE::DetectionOutputCodeTypeAttr::get(_ctx, IE::DetectionOutputCodeType::CORNER_SIZE);
    }

    VPUX_THROW("Unknown DetectionOutput code_type");
}

IE::DetectionOutputAttr NGraphImporter::importDetectionOutputAttrs(const ov::op::v0::DetectionOutput::Attributes& val) {
    const auto numClassesAttr = getIntAttr(_ctx, val.num_classes);
    const auto backgroundLabelIdAttr = getIntAttr(_ctx, val.background_label_id);
    const auto topKAttr = getIntAttr(_ctx, val.top_k);

    const auto varianceEncodedInTargetAttr = mlir::BoolAttr::get(_ctx, val.variance_encoded_in_target);

    const auto keepTopKAttr = getIntArrayAttr(_ctx, val.keep_top_k);
    const auto codeTypeAttr = importDetectionOutputCodeType(val.code_type);

    const auto shareLocationAttr = mlir::BoolAttr::get(_ctx, val.share_location);

    const auto nmsThresholdAttr = getFPAttr(_ctx, val.nms_threshold);
    const auto confidenceThresholdAttr = getFPAttr(_ctx, val.confidence_threshold);

    const auto clipAfterNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_after_nms);
    const auto clipBeforeNmsAttr = mlir::BoolAttr::get(_ctx, val.clip_before_nms);
    const auto decreaseLabel_idAttr = mlir::BoolAttr::get(_ctx, val.decrease_label_id);
    const auto normalizedAttr = mlir::BoolAttr::get(_ctx, val.normalized);

    const auto inputHeightAttr = getIntAttr(_ctx, val.input_height);
    const auto inputWidthAttr = getIntAttr(_ctx, val.input_width);

    const auto objectnessScoreAttr = getFPAttr(_ctx, val.objectness_score);

    return IE::DetectionOutputAttr::get(_ctx, numClassesAttr, backgroundLabelIdAttr, topKAttr,
                                        varianceEncodedInTargetAttr, keepTopKAttr, codeTypeAttr, shareLocationAttr,
                                        nmsThresholdAttr, confidenceThresholdAttr, clipAfterNmsAttr, clipBeforeNmsAttr,
                                        decreaseLabel_idAttr, normalizedAttr, inputHeightAttr, inputWidthAttr,
                                        objectnessScoreAttr);
}

IE::ExperimentalDetectronROIFeatureExtractorAttr NGraphImporter::importExpDetectronROIFeatureExtractAttrs(
        const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& val) {
    const auto outputSizeAttr = getIntAttr(_ctx, val.output_size);
    const auto samplingRatioAttr = getIntAttr(_ctx, val.sampling_ratio);
    const auto pyramidScalesAttr = getIntArrayAttr(_ctx, val.pyramid_scales);
    const auto alignedAttr = mlir::BoolAttr::get(_ctx, val.aligned);

    return IE::ExperimentalDetectronROIFeatureExtractorAttr::get(_ctx, outputSizeAttr, samplingRatioAttr, alignedAttr,
                                                                 pyramidScalesAttr);
}

IE::ROIPoolingMethodAttr NGraphImporter::importROIPoolingMethod(const std::string& method) {
    IE::ROIPoolingMethodAttr attr;
    if (method == "max") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::MAX);
    } else if (method == "bilinear") {
        attr = IE::ROIPoolingMethodAttr::get(_ctx, IE::ROIPoolingMethod::BILINEAR);
    } else {
        VPUX_THROW("Unknown ROIPoolingMethod");
    }
    return attr;
}

IE::PSROIPoolingModeAttr NGraphImporter::importPSROIPoolingMode(const std::string& mode) {
    if (mode == "average") {
        return IE::PSROIPoolingModeAttr::get(_ctx, IE::PSROIPoolingMode::AVERAGE);
    } else if (mode == "bilinear") {
        return IE::PSROIPoolingModeAttr::get(_ctx, IE::PSROIPoolingMode::BILINEAR);
    }

    VPUX_THROW("Unknown PSROIPoolingMode: {0}", mode);
}

IE::ROIAlignMethodAttr NGraphImporter::importROIAlignMethod(const ov::op::v9::ROIAlign::PoolingMode& mode) {
    IE::ROIAlignMethodAttr attr;
    if (mode == ov::op::v9::ROIAlign::PoolingMode::AVG) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::AVG);
    } else if (mode == ov::op::v9::ROIAlign::PoolingMode::MAX) {
        attr = IE::ROIAlignMethodAttr::get(_ctx, IE::ROIAlignMethod::MAX);
    } else {
        VPUX_THROW("Unknown ROIAlignMethod");
    }
    return attr;
}

IE::ROIAlignAlignedMethodAttr NGraphImporter::importROIAlignAlignedMethod(
        const ov::op::v9::ROIAlign::AlignedMode& mode) {
    IE::ROIAlignAlignedMethodAttr attr;
    if (mode == ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::ASYMMETRIC);
    } else if (mode == ov::op::v9::ROIAlign::AlignedMode::HALF_PIXEL) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::HALF_PIXEL);
    } else if (mode == ov::op::v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN) {
        attr = IE::ROIAlignAlignedMethodAttr::get(_ctx, IE::ROIAlignAlignedMethod::HALF_PIXEL_FOR_NN);
    } else {
        VPUX_THROW("Unknown ROIAlignAlignedMethod");
    }
    return attr;
}

IE::RNNSequenceDirectionAttr NGraphImporter::importRNNSequenceDirection(const ov::op::RecurrentSequenceDirection val) {
    IE::RNNSequenceDirectionAttr attr;
    if (val == ov::op::RecurrentSequenceDirection::FORWARD) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::FORWARD);
    } else if (val == ov::op::RecurrentSequenceDirection::REVERSE) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::REVERSE);
    } else if (val == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        attr = IE::RNNSequenceDirectionAttr::get(_ctx, IE::RNNSequenceDirection::BIDIRECTIONAL);
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
    }
    return attr;
}

IE::BoxEncodingTypeAttr NGraphImporter::importBoxEncodingType(const int box_encoding_type) {
    return box_encoding_type ? IE::BoxEncodingTypeAttr::get(_ctx, IE::BoxEncodingType::CENTER)
                             : IE::BoxEncodingTypeAttr::get(_ctx, IE::BoxEncodingType::CORNER);
}

IE::PadModeAttr NGraphImporter::importPadMode(const ov::op::PadMode val) {
    IE::PadModeAttr attr;
    switch (val) {
    case ov::op::PadMode::CONSTANT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::CONSTANT);
        break;
    case ov::op::PadMode::EDGE:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::EDGE);
        break;
    case ov::op::PadMode::REFLECT:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::REFLECT);
        break;
    case ov::op::PadMode::SYMMETRIC:
        attr = IE::PadModeAttr::get(_ctx, IE::PadMode::SYMMETRIC);
        break;
    default:
        VPUX_THROW("Unknown PadMode");
    }
    return attr;
}

IE::RoundModeAttr NGraphImporter::importRoundMode(const ov::op::v5::Round::RoundMode val) {
    IE::RoundModeAttr attr;
    switch (val) {
    case ov::op::v5::Round::RoundMode::HALF_TO_EVEN:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_TO_EVEN);
        break;
    case ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
        attr = IE::RoundModeAttr::get(_ctx, IE::RoundMode::HALF_AWAY_FROM_ZERO);
        break;
    default:
        VPUX_THROW("Unknown RoundMode");
    }
    return attr;
}

IE::DepthToSpaceModeAttr NGraphImporter::importDepthToSpaceMode(const ov::op::v0::DepthToSpace::DepthToSpaceMode val) {
    IE::DepthToSpaceModeAttr attr;
    switch (val) {
    case ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::BLOCKS_FIRST);
        break;
    case ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
        attr = IE::DepthToSpaceModeAttr::get(_ctx, IE::DepthToSpaceMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown DepthToSpace Mode");
    }
    return attr;
}

IE::SpaceToDepthModeAttr NGraphImporter::importSpaceToDepthMode(const ov::op::v0::SpaceToDepth::SpaceToDepthMode val) {
    IE::SpaceToDepthModeAttr attr;
    switch (val) {
    case ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::BLOCKS_FIRST);
        break;
    case ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
        attr = IE::SpaceToDepthModeAttr::get(_ctx, IE::SpaceToDepthMode::DEPTH_FIRST);
        break;
    default:
        VPUX_THROW("Unknown SpaceToDepthMode");
    }
    return attr;
}

IE::PadTypeAttr NGraphImporter::importPadType(ov::op::PadType padding) {
    switch (padding) {
    case ov::op::PadType::SAME_UPPER:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::SAME_UPPER);
    case ov::op::PadType::SAME_LOWER:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::SAME_LOWER);
    case ov::op::PadType::VALID:
        return IE::PadTypeAttr::get(_ctx, IE::PadType::VALID);
    default:
        VPUX_THROW("Unknown PadType {0}", static_cast<int>(padding));
    }
}

IE::SliceInputPortMapAttr NGraphImporter::importSliceInputPortMapAttr(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>& desc,
        const mlir::Value input, const int64_t numIter) {
    auto shape = getShape(input).raw();
    int64_t axisInt = desc->m_axis < 0 ? desc->m_axis + shape.size() : desc->m_axis;
    int64_t axisLen = shape[axisInt];

    const auto externalPortId = getIntAttr(ctx, desc->m_input_index);
    const auto internalLayerId = getIntAttr(ctx, desc->m_body_parameter_index);
    const auto axis = getIntAttr(ctx, axisInt);
    auto startInt = desc->m_start < 0 ? desc->m_start + axisLen : desc->m_start;
    const auto start = getIntAttr(ctx, startInt);
    auto strideInt = desc->m_stride;
    const auto stride = getIntAttr(ctx, strideInt);
    auto endInt = desc->m_end < 0 ? desc->m_end + axisLen : desc->m_end;
    const auto end = getIntAttr(ctx, endInt);

    VPUX_THROW_UNLESS((axisInt >= 0) && (startInt >= 0) && (endInt >= 0),
                      "Wrong start `{0}`, end `{1}`, axis `{2}` value, out of range", desc->m_start, desc->m_end,
                      desc->m_axis);

    _log.trace("axisInt {0} axisLen {1} startInt {2} endInt {3}", axisInt, axisLen, startInt, endInt);
    VPUX_THROW_UNLESS((endInt - startInt) * strideInt >= 0,
                      "Wrong start `{0}`, end `{1}`, stride `{2}` combination, shape `{3}`, axis `{4}`", desc->m_start,
                      desc->m_end, desc->m_stride, shape, desc->m_axis);

    int64_t partSizeInt = numIter > 0 ? max(endInt + 1 - startInt, startInt - endInt + 1) / numIter : 1;
    VPUX_THROW_UNLESS(partSizeInt > 0,
                      "num_iterations should not be more than axis length, actual num_iterations {0}, axis length {1}",
                      numIter, max(endInt + 1 - startInt, startInt - endInt + 1));
    const auto partSize = getIntAttr(ctx, partSizeInt);
    return IE::SliceInputPortMapAttr::get(ctx, externalPortId, internalLayerId, axis, start, stride, partSize, end);
}

IE::InvariantInputPortMapAttr NGraphImporter::importInvariantInputPortMapAttr(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>& desc) {
    const auto externalPortId = getIntAttr(ctx, desc->m_input_index);
    const auto internalLayerId = getIntAttr(ctx, desc->m_body_parameter_index);
    return IE::InvariantInputPortMapAttr::get(ctx, externalPortId, internalLayerId);
}

IE::MergedInputPortMapAttr NGraphImporter::importMergedInputPortMapAttr(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>& desc) {
    const auto externalPortId = getIntAttr(ctx, desc->m_input_index);
    const auto internalLayerId = getIntAttr(ctx, desc->m_body_parameter_index);
    const auto bodyInputIndex = getIntAttr(ctx, desc->m_body_value_index);
    return IE::MergedInputPortMapAttr::get(ctx, externalPortId, internalLayerId, bodyInputIndex);
}

IE::ConcatOutputPortMapAttr NGraphImporter::importConcatOutputPortMapAttr(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>& desc) {
    const auto externalPortId = getIntAttr(ctx, desc->m_output_index);
    const auto internalLayerId = getIntAttr(ctx, desc->m_body_value_index);
    const auto axis = getIntAttr(ctx, desc->m_axis);
    const auto start = getIntAttr(ctx, desc->m_start);
    const auto stride = getIntAttr(ctx, desc->m_stride);
    const auto partSize = getIntAttr(ctx, desc->m_part_size);
    const auto end = getIntAttr(ctx, desc->m_end);
    return IE::ConcatOutputPortMapAttr::get(ctx, externalPortId, internalLayerId, axis, start, stride, partSize, end);
}

IE::InvariantOutputPortMapAttr NGraphImporter::importBodyOutputPortMapAttr(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>& desc) {
    const auto externalPortId = getIntAttr(ctx, desc->m_output_index);
    const auto internalLayerId = getIntAttr(ctx, desc->m_body_value_index);
    const auto iterations = getIntAttr(ctx, desc->m_iteration);
    return IE::InvariantOutputPortMapAttr::get(ctx, externalPortId, internalLayerId, iterations);
}

IE::DeformablePSROIPoolingModeAttr NGraphImporter::importDeformablePSROIPoolingMode(const std::string& mode) {
    if (mode == "average") {
        return IE::DeformablePSROIPoolingModeAttr::get(_ctx, IE::DeformablePSROIPoolingMode::AVERAGE);
    } else if (mode == "bilinear_deformable") {
        return IE::DeformablePSROIPoolingModeAttr::get(_ctx, IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE);
    }

    VPUX_THROW("Unknown DeformablePSROIPoolingMode: {0}", mode);
}

IE::ScatterElementsUpdateReductionTypeAttr NGraphImporter::importScatterElementsUpdateReductionType(
        ov::op::v12::ScatterElementsUpdate::Reduction val) {
    switch (val) {
    case ov::op::v12::ScatterElementsUpdate::Reduction::NONE:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::NONE);
    case ov::op::v12::ScatterElementsUpdate::Reduction::SUM:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::SUM);
    case ov::op::v12::ScatterElementsUpdate::Reduction::PROD:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::PROD);
    case ov::op::v12::ScatterElementsUpdate::Reduction::MIN:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::MIN);
    case ov::op::v12::ScatterElementsUpdate::Reduction::MAX:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::MAX);
    case ov::op::v12::ScatterElementsUpdate::Reduction::MEAN:
        return IE::ScatterElementsUpdateReductionTypeAttr::get(_ctx, IE::ScatterElementsUpdateReductionType::MEAN);
    default:
        VPUX_THROW("Unknown ScatterElementsUpdate reduction type");
    }
}

mlir::RankedTensorType importUserTensor(mlir::MLIRContext* ctx, const ov::descriptor::Tensor& tensor) {
    const auto& partialShape = tensor.get_partial_shape();

    const auto importedShape = importShape(partialShape);
    const vpux::Shape shape(importedShape.begin(), importedShape.end());
    const auto precision = importPrecision(ctx, tensor.get_element_type());
    auto importedTensor = getTensorType(shape, precision, DimsOrder::fromNumDims(shape.size()), nullptr);

    if (partialShape.is_dynamic()) {
        ov::PartialShape boundedOutShape;
        try {
            std::transform(partialShape.begin(), partialShape.end(), std::back_inserter(boundedOutShape), toBoundedDim);
        } catch (...) {
            // FIXME(E#151586) remove this try-catch block
            Logger::global().warning("Found partialShape without bounds: {0}", partialShape.to_string());
            return importedTensor;
        }

        auto ndType = mlir::cast<NDTypeInterface>(importedTensor);
        const auto boundShape = boundedOutShape.to_shape();
        const Bounds bounds(boundShape.begin(), boundShape.end());
        ndType = Core::BoundedTensorType::get(ndType, bounds);

        return mlir::cast<mlir::RankedTensorType>(ndType);
    }

    return importedTensor;
}

//
// runNGraphPasses
//

static bool skipKeepConstAndDecompressionForNode(const std::shared_ptr<const ov::Node>& node) {
    if (ov::shape_size(node->get_input_shape(0)) <= STATIC_RANK_MAX_DIMS) {
        return true;
    }

    // when calling addCommonOptimizationsPasses() certain pattern won't match anymore if this pass is
    // applied to them and Const->Convert won't get folded
    auto nextNode = node->get_output_target_inputs(0).begin()->get_node();
    if (ov::is_type<ov::op::v1::Multiply>(nextNode)) {
        auto multiplyInputNode = nextNode->get_input_node_shared_ptr(0);
        // PReluFusion pattern won't match if the Convert node doesn't get folded
        // so Relu->Multiply won't become PRelu
        if (ov::is_type<ov::op::v0::Relu>(multiplyInputNode)) {
            return true;
        }

        // if ConvertSubtractWithConstant is applied, Subtract will become Add
        // then AddMultiplyFusion (from LinOpSequenceFusion pass) will be applied, if Multiply has
        // a Constant as an input
        if (ov::is_type<ov::op::v1::Subtract>(std::move(multiplyInputNode))) {
            return true;
        }

        // AddMultiplyFusion won't match if the Convert node doesn't get folded, so Add needs to have
        // a Constant as an input
        auto multiplyOutputNode = nextNode->get_output_target_inputs(0).begin()->get_node();
        if (ov::is_type<ov::op::v1::Add>(multiplyOutputNode)) {
            return true;
        }

        // in the subgraph Multiply -> Concat -> Slice -> Multiply
        // Slice will get eliminated during StridedSliceOptimization pass
        // Concat will get eliminated during EliminateConcat pass
        // MultiplyMultiplyFusion needs both Multiply nodes to have a Constant as an input in order to
        // replace them with only one Multiply
        if (ov::is_type<ov::op::v0::Concat>(multiplyOutputNode)) {
            auto concatOutputs = multiplyOutputNode->get_output_target_inputs(0);
            if (concatOutputs.size() > 1) {
                auto concatOutputNode = std::next(concatOutputs.begin(), 1)->get_node();
                if (ov::is_type<ov::op::v8::Slice>(concatOutputNode)) {
                    auto sliceOutputNode = concatOutputNode->get_output_target_inputs(0).begin()->get_node();
                    if (ov::is_type<ov::op::v1::Multiply>(sliceOutputNode)) {
                        return true;
                    }
                }
            }
        }
    }

    if (ov::is_type<ov::op::v1::Convolution>(nextNode)) {
        auto convolutionInputNode = nextNode->get_input_node_shared_ptr(0);
        // For Transpose->Convolution, Convolution should have a Constant as an input, in case an operation
        // like Multiply sinks through Transpose during one of the optimizations
        // in addCommonOptimizationsPasses() so then the Multiply->Convolution pattern will be matched by
        // MultiplyConvolutionFusion
        if (ov::is_type<ov::op::v1::Transpose>(convolutionInputNode)) {
            return true;
        }

        // MultiplyConvolutionFusion pattern won't match if Convert doesn't get folded, so Convolution
        // needs to have a Constant as an input
        if (ov::is_type<ov::op::v1::Multiply>(convolutionInputNode)) {
            return true;
        }

        // in the case of Multiply -> Add -> Convolution, when EliminateEltwise pass is applied, the Add
        // operations will be eliminated and MultiplyConvolutionFusion won't be applied if Convolution
        // doesn't have a Constant as an input
        if (ov::is_type<ov::op::v1::Add>(convolutionInputNode)) {
            auto addInputNode = convolutionInputNode->get_input_node_shared_ptr(0);
            if (ov::is_type<ov::op::v1::Multiply>(std::move(addInputNode))) {
                return true;
            }
        }

        // in case of Divide -> Convolution, Divide will become Multiply after ConvertDivideWithConstant
        // and Convolution needs to have a Constant as an input for MultiplyConvolutionFusion to be applied properly
        if (ov::is_type<ov::op::v1::Divide>(convolutionInputNode)) {
            return true;
        }

        // DilatedConvolutionConverter will replace SpaceToBatch -> Convolution -> BatchToSpace to a single
        // Convolution node this is later followed by ConvolutionMultiplyFusion so this Convolution needs to
        // have a Constant as an input if the previosly mentioned subgraph is followed by Multiply
        auto convolutionOutputNode = nextNode->get_output_target_inputs(0).begin()->get_node();
        if (ov::is_type<ov::op::v1::SpaceToBatch>(std::move(convolutionInputNode)) &&
            ov::is_type<ov::op::v1::BatchToSpace>(convolutionOutputNode)) {
            auto batchToSpaceOutputNode = convolutionOutputNode->get_output_target_inputs(0).begin()->get_node();
            if (ov::is_type<ov::op::v1::Multiply>(batchToSpaceOutputNode)) {
                return true;
            }
        }

        // ConvolutionMultiplyFusion pass will make Multiply the new input of Convolution
        // if Convolution doesn't have a Constant as the original input, neither will Multiply
        // and it won't be able to get folded in and eliminated later on
        if (ov::is_type<ov::op::v1::Multiply>(std::move(convolutionOutputNode))) {
            return true;
        }
    }

    if (ov::is_type<ov::op::v1::Subtract>(nextNode)) {
        auto substractOutputNode = nextNode->get_output_target_inputs(0).begin()->get_node();
        // if ConvertSubtractWithConstant is applied, Subtract will become Add only if Subtract has a
        // Constant as an input, then AddMultiplyFusion will be applied
        if (ov::is_type<ov::op::v1::Multiply>(std::move(substractOutputNode))) {
            return true;
        }
    }

    if (ov::is_type<ov::op::v1::Add>(nextNode)) {
        auto addInputNode = nextNode->get_input_node_shared_ptr(0);
        // AddMultiplyFusion pattern won't apply if Add does't have a Constant as an input
        if (ov::is_type<ov::op::v1::Multiply>(std::move(addInputNode))) {
            return true;
        }
    }

    if (ov::is_type<ov::op::v8::Gather>(nextNode)) {
        auto gatherOutputNode = nextNode->get_output_target_inputs(0).begin()->get_node();
        // Gather->Add won't be folded if Gather doesn't have a Constant as an input
        if (ov::is_type<ov::op::v1::Add>(std::move(gatherOutputNode))) {
            return true;
        }
    }

    if (ov::is_type<ov::op::v11::Interpolate>(nextNode)) {
        auto interpolateOutputNode = nextNode->get_output_target_inputs(0).begin()->get_node();
        // Interpolate->Transpose->Reshape won't be folded if Interpolate doesn't have a Constant as an input
        if (ov::is_type<ov::op::v1::Transpose>(interpolateOutputNode)) {
            auto transposeOutputNode = interpolateOutputNode->get_output_target_inputs(0).begin()->get_node();
            if (ov::is_type<ov::op::v1::Reshape>(std::move(transposeOutputNode))) {
                return true;
            }
        }
    }

    return false;
}

static void addCommonOptimizationsPasses(ov::pass::Manager& manager) {
    // MOCTransformations contain StridedSliceOptimization transformation,
    // so we must call SliceToStridedSlice before MOCTransformations call
    manager.register_pass<ov::pass::SliceToStridedSlice>(true);
    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    manager.register_pass<ov::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::PadFusionConvolution>();
    pass_config->disable<ov::pass::PadFusionGroupConvolution>();
    pass_config->disable<ov::pass::MVNFusionWithConstantsInside>();
    pass_config->disable<ov::pass::PullThroughReduce>();
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();
    pass_config->disable<ov::pass::MulFakeQuantizeFusion>();

    // NMS conversion passes
    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();

    auto static_shape = manager.register_pass<ov::pass::GraphRewrite>();
    static_shape->add_matcher<ov::pass::ConvertNMS9ToNMSIEInternal>();
    static_shape->set_name("ov::pass::CommonStaticShape");

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::RMSFusion>();
    common_fusions->set_name("ov::pass::CommonFusions");

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
    decomp->add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::ConvertConvertLike>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::DropoutWithRandomUniformReplacer>();

    bool enableDecompositionForSDPA = true;
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("NPU_DECOMPOSE_SDPA", enableDecompositionForSDPA);
#endif
    decomp->add_matcher<ov::pass::GroupQueryAttentionDecomposition>();
    if (enableDecompositionForSDPA) {
        decomp->add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    decomp->add_matcher<ov::pass::GroupNormalizationDecomposition>();
    decomp->set_name("ov::pass::CommonDecompositions");
    manager.register_pass<ov::pass::SliceToStridedSlice>(true);

    // CF is required after all decompositions
    manager.register_pass<ov::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ov::pass::LinOpSequenceFusion>();
    manager.register_pass<ov::pass::UnrollIf>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();

    auto maxpool_conversion = manager.register_pass<ov::pass::GraphRewrite>();
    maxpool_conversion->add_matcher<ov::pass::ConvertMaxPool14ToMaxPool8>();
    maxpool_conversion->set_name("ov::pass::ConditionalMaxPool14ToMaxPool8Conversion");

    pass_config->set_callback<ov::pass::ConvertMaxPool14ToMaxPool8>(
            [](const std::shared_ptr<const ov::Node>& node) -> bool {
                if (auto maxpool = ov::as_type_ptr<const ov::opset14::MaxPool>(node)) {
                    return maxpool->get_rounding_type() ==
                           ov::op::RoundingType::CEIL_TORCH;  // Skip conversion if rounding type is CEIL_TORCH
                }
                return false;  // Don't skip for other rounding types
            });

    manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    manager.register_pass<ov::pass::ConvertShapeOf3>();

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ov::pass::StridesOptimization>();
}

void NGraphPasses::runNGraphPasses(const std::shared_ptr<ov::Model>& netGraph, mlir::TimingScope& rootTiming,
                                   bool isWeightsSeparationPath) {
    auto scopeTiming = rootTiming.nest("Common nGraph passes");

    ov::pass::Manager manager;
    auto passConfig = manager.get_pass_config();
    manager.register_pass<ov::pass::InitNodeInfo>();

    // Dequantization of the constants with types specified here will not be folded. If folding occurs the compiler will
    // think weights are FP16/FP32 resulting in high-precision execution.
    ov::element::TypeVector decompression_precisions{ov::element::u4,     ov::element::i4,    ov::element::nf4,
                                                     ov::element::u8,     ov::element::i8,    ov::element::f8e4m3,
                                                     ov::element::f8e5m2, ov::element::f8e8m0};
    manager.register_pass<ov::pass::MarkDequantization>(decompression_precisions, /*fold_subtract_const=*/true);
    // Prevent 2bit zero-point constants from being folded by OV.
    // We need to use the sub-byte unpacking order from compiler.
    ov::element::TypeVector decompression_precisions_keep_lp_zp{ov::element::u2};
    manager.register_pass<ov::pass::MarkDequantization>(decompression_precisions_keep_lp_zp,
                                                        /*fold_subtract_const=*/false);
    manager.register_pass<ov::pass::KeepConstPrecision>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::KeepConstAndDecompression>();
    passConfig->set_callback<ov::pass::KeepConstAndDecompression>(
            [&](const std::shared_ptr<const ov::Node>& node) -> bool {
                if (isWeightsSeparationPath) {
                    return false;
                }

                return skipKeepConstAndDecompressionForNode(node);
            });
    manager.register_pass<ov::pass::SharedOpOptimization>();
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    manager.register_pass<ov::pass::ConstantFolding>();
    addCommonOptimizationsPasses(manager);

    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (const auto serializeCanonicalModel = std::getenv("NPU_SERIALIZE_CANONICAL_MODEL")) {
        if (intel_npu::envVarStrToBool("NPU_SERIALIZE_CANONICAL_MODEL", serializeCanonicalModel)) {
            const std::string graphName = netGraph->get_friendly_name();
            manager.register_pass<ov::pass::Serialize>(graphName + "_canonical.xml", graphName + "_canonical.bin");
        }
    }
#endif

    manager.run_passes(netGraph);
}

//
// addNetworkInfoOp
//

mlir::StringAttr createPrimaryNameAttr(mlir::MLIRContext* ctx, const ov::Node& node) {
    auto primaryName = ov::op::util::is_output(&node) ? ov::op::util::get_ie_output_name(node.input_value(0))
                                                      : node.get_friendly_name();
    return mlir::StringAttr::get(ctx, primaryName);
}

mlir::TypeAttr createOriginalShapeAttr(mlir::MLIRContext* ctx, const ov::descriptor::Tensor& actualTensor,
                                       const ov::descriptor::Tensor& originalTensor) {
    return (actualTensor.get_partial_shape() != originalTensor.get_partial_shape())
                   ? mlir::TypeAttr::get(importUserTensor(ctx, originalTensor))
                   : nullptr;
}

mlir::StringAttr createInputNameAttr(mlir::MLIRContext* ctx, const ov::Node& node) {
    if (ov::op::util::is_output(&node) || ov::op::util::is_parameter(&node)) {
        auto inputName =
                ov::op::util::get_ie_output_name(ov::op::util::is_output(&node) ? node.input_value(0) : node.output(0));
        return mlir::StringAttr::get(ctx, inputName);
    }
    return nullptr;
}

mlir::ArrayAttr createTensorNamesAttr(mlir::MLIRContext* ctx, const ov::Node& node) {
    VPUX_THROW_UNLESS(node.get_output_size() == 1, "{0} has more than 1 output tensors: {1}",
                      ov::op::util::is_output(&node) ? ov::op::util::get_ie_output_name(node.input_value(0))
                                                     : node.get_friendly_name(),
                      node.get_output_size());
    auto tensorNames = to_small_vector(node.get_output_tensor(0).get_names());
    llvm::sort(tensorNames);
    SmallVector<mlir::Attribute> tensorNamesVector(tensorNames.size());
    for (auto tensorName : tensorNames | indexed) {
        tensorNamesVector[tensorName.index()] = mlir::StringAttr::get(ctx, tensorName.value());
    }
    return !tensorNamesVector.empty() ? mlir::ArrayAttr::get(ctx, tensorNamesVector) : nullptr;
}

void createDataInfoOp(mlir::OpBuilder infoBuilder, mlir::MLIRContext* ctx, const ov::Node& node,
                      const ov::Node& originalNode) {
    const auto primaryNameAttr = createPrimaryNameAttr(ctx, originalNode);
    const auto loc = createLayerLocation(ctx, node.get_friendly_name(), originalNode.description());

    const auto userTypeAttr = mlir::TypeAttr::get(importUserTensor(ctx, node.get_output_tensor(0)));

    const auto originalShapeAttr =
            createOriginalShapeAttr(ctx, node.get_output_tensor(0), originalNode.get_output_tensor(0));

    const auto friendlyNameAttr = (originalNode.get_friendly_name() != primaryNameAttr.str())
                                          ? mlir::StringAttr::get(ctx, originalNode.get_friendly_name())
                                          : nullptr;

    const auto inputNameAttr = (createInputNameAttr(ctx, originalNode) != primaryNameAttr)
                                       ? createInputNameAttr(ctx, originalNode)
                                       : nullptr;

    const auto tensorNamesAttr = createTensorNamesAttr(ctx, originalNode);

    infoBuilder.create<net::DataInfoOp>(loc, primaryNameAttr, userTypeAttr,
                                        /*OptionalAttr originalShape*/ originalShapeAttr,
                                        /*OptionalAttr friendlyName*/ friendlyNameAttr,
                                        /*OptionalAttr inputName*/ inputNameAttr,
                                        /*OptionalAttr tensorNames*/ tensorNamesAttr,
                                        /*profilingSectionsCount=*/0);
}

void addNetworkInfoOp(mlir::OpBuilder& builder, mlir::FlatSymbolRefAttr mainFuncName,
                      const std::shared_ptr<ov::Model>& model,
                      const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                      const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                      mlir::TimingScope& rootTiming, bool enableProfiling) {
    auto scopeTiming = rootTiming.nest("Add NetworkInfo Operation");

    const auto parameters = model->get_parameters();
    const auto results = model->get_results();

    // Used to set interval for dimension of dynamic shape of Range.
    for (const auto& result : results) {
        auto parentNode = result->get_input_source_output(0).get_node_shared_ptr();
        if (auto range = ov::as_type_ptr<ov::op::v4::Range>(parentNode)) {
            auto partialShape = range->get_output_partial_shape(0);
            if (partialShape.is_dynamic()) {
                auto outputShape = ov::PartialShape{{0, 1}};
                auto& dim = outputShape[0];
                auto bounds = ov::Dimension(static_cast<int64_t>(RANGEBOUND));
                dim *= bounds;
                range->set_output_type(0, range->get_output_type(), outputShape);
            }
        }
    }

    auto* ctx = builder.getContext();

    auto netInfo = builder.create<net::NetworkInfoOp>(mlir::UnknownLoc::get(ctx), mainFuncName, enableProfiling);
    net::setupSections(netInfo, enableProfiling);

    VPUX_THROW_UNLESS(parameters.size() == originalParameters.size(),
                      "Number of parameters are different between actual ({0}) and original model ({1})",
                      parameters.size(), originalParameters.size());
    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&netInfo.getInputsInfo().front(), builder.getListener());
    for (size_t i = 0; i < parameters.size(); i++) {
        createDataInfoOp(inputsInfoBuilder, ctx, *parameters[i], *originalParameters[i]);
    }

    VPUX_THROW_UNLESS(results.size() == originalResults.size(),
                      "Number of results are different between actual ({0}) and original model ({1})", results.size(),
                      originalResults.size());
    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&netInfo.getOutputsInfo().front(), builder.getListener());
    for (size_t i = 0; i < results.size(); i++) {
        createDataInfoOp(outputsInfoBuilder, ctx, *results[i], *originalResults[i]);
    }
}

//
// addSparsityStatistics
//

void addSparsityStatistics(const std::shared_ptr<ov::Model>& model, mlir::ModuleOp module, Logger log) {
    using RTMap = std::map<std::string, ov::Any>;

    auto maybeRt = model->get_rt_info();
    if (maybeRt.empty()) {
        log.trace("Missed RT info");
        return;
    }
    log.trace("Found RT info");
    if (!model->has_rt_info(NGRAPH_ACT_SPARSITY_STATS_KEY)) {
        return;
    }
    log.trace("Found activation statistics");
    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    auto* ctx = builder.getContext();

    auto statsOp = builder.create<net::SparsityStatisticsOp>(mlir::UnknownLoc::get(ctx));
    statsOp.getSparsityInfo().emplaceBlock();

    auto statsBuilder = mlir::OpBuilder::atBlockBegin(&statsOp.getSparsityInfo().front(), builder.getListener());

    auto actSparsityStats = model->get_rt_info<RTMap>(NGRAPH_ACT_SPARSITY_STATS_KEY);
    for (const auto& x : actSparsityStats) {
        const auto key = x.first;
        const auto nodeName = getSparsityStatsFieldChecked<std::string>(model, key, "node_name");
        const auto portId = getSparsityStatsFieldChecked<int>(model, key, "port_id");
        const auto ratio = getSparsityStatsFieldChecked<double>(model, key, "statistic");

        const auto nameAttr = mlir::StringAttr::get(ctx, nodeName);
        const auto inputAttr = getIntAttr(ctx, portId);
        const auto ratioAttr = getFPAttr(ctx, ratio);
        statsBuilder.create<net::SparsityInfoOp>(mlir::UnknownLoc::get(ctx), nameAttr, inputAttr, ratioAttr);
    }
}

//
// importNetwork
//

// Applies upper bounds to the model, gets rid of the dynamism
void dynamicToStaticShape(const std::shared_ptr<ov::Model>& model) {
    for (const auto& input : model->get_parameters()) {
        if (input->get_partial_shape().is_dynamic()) {
            const auto& inputPartialShape = input->get_partial_shape();
            ov::PartialShape newShape;
            std::transform(inputPartialShape.begin(), inputPartialShape.end(), std::back_inserter(newShape),
                           toBoundedDim);
            input->set_partial_shape(newShape);
        }
    }
}

mlir::OwningOpRef<mlir::ModuleOp> vpux::IE::importNetwork(
        mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
        const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
        const std::vector<std::shared_ptr<const ov::Node>>& originalResults, mlir::TimingScope& rootTiming,
        const vpux::IE::ImportNetworkConfig& importCfg, Logger log) {
    log.setName("IE::FrontEnd::importNetwork");

    log.trace("Load IE::FrontEnd dependent Dialects");
    // BuiltInDialect with BuiltInTypes will be initialized when MLIRContext is initialized.
    // Therefore, we should also explicitly initialize QuantileFloatDialect asap.
    ctx->loadDialect<vpux::type::QuantileFloatDialect>();
    ctx->loadDialect<IE::IEDialect>();

    if (importCfg.dynamicShapeToStatic && model->is_dynamic()) {
        dynamicToStaticShape(model);
    }

    log.trace("Run common nGraph passes");
    NGraphPasses::runNGraphPasses(model, rootTiming, importCfg.enableWeightsSeparationPath);

    const auto moduleLoc = IE::createLayerLocation(ctx, "module", "Module");
    auto module = mlir::ModuleOp::create(moduleLoc, StringRef(model->get_friendly_name()));
    addSparsityStatistics(model, module, log);
    const auto mainFuncName = mlir::FlatSymbolRefAttr::get(ctx, "main");

    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    log.trace("Add NetworkInfo Operation");
    addNetworkInfoOp(builder, mainFuncName, model, originalParameters, originalResults, rootTiming,
                     importCfg.enableProfiling);

    log.trace("Import nGraph function");

    NGraphImporter importer(ctx, model, importCfg.sharedConstants, log);
    importer.buildMainFunc(builder, mainFuncName.getValue(), rootTiming, importCfg.stubLayers,
                           importCfg.dynamicShapeToStatic);

    log.trace("Validate MLIR module");
    auto finalTiming = rootTiming.nest("Validate MLIR module");
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)), "Failed to create a valid MLIR module for the IR model");

    return module;
}

//
// Build OV Nodes helper functions
//

bool isIR10(const ov::Model& model) {
    const auto& rtInfo = model.get_rt_info();
    const auto it = rtInfo.find("version");
    if (it != rtInfo.end()) {
        const int64_t irVersion = it->second.as<int64_t>();
        return irVersion == 10;
    }
    return false;
}

std::vector<std::shared_ptr<const ov::Node>> vpux::IE::buildOVParams(const std::shared_ptr<const ov::Model>& model) {
    const auto params = model->get_parameters();
    std::vector<std::shared_ptr<const ov::Node>> constParams(params.size());
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& param : params | indexed) {
        auto newParam = ov::as_type_ptr<ov::op::v0::Parameter>(param.value()->copy_with_new_inputs({}));
        newParam->set_friendly_name(param.value()->get_friendly_name());
        if (addOpNames) {
            newParam->output(0).get_tensor().add_names({newParam->get_friendly_name()});
        }
        newParam->validate_and_infer_types();
        constParams[param.index()] = newParam;
    }

    return constParams;
}

std::vector<std::shared_ptr<const ov::Node>> vpux::IE::buildOVResults(const std::shared_ptr<const ov::Model>& model) {
    const auto results = model->get_results();
    std::vector<std::shared_ptr<const ov::Node>> constResults(results.size());
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& result : results | indexed) {
        auto fakeParam = std::make_shared<ov::op::v0::Parameter>(result.value()->get_output_element_type(0),
                                                                 result.value()->get_output_partial_shape(0));
        const std::string paramName = ov::op::util::create_ie_output_name(result.value()->input_value(0));
        fakeParam->set_friendly_name(paramName);
        fakeParam->validate_and_infer_types();
        auto newResult = result.value()->copy_with_new_inputs({fakeParam});
        newResult->set_friendly_name(result.value()->get_friendly_name());
        if (addOpNames) {
            newResult->output(0).get_tensor().add_names({fakeParam->get_friendly_name()});
        }
        constResults[result.index()] = newResult;
    }

    return constResults;
}
