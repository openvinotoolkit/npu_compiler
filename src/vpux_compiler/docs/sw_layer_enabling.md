# MLIR software layer enabling steps
- [MLIR software layer enabling steps](#mlir-software-layer-enabling-steps)
- [Introduction](#introduction)
- [Debugging tips and tricks](#debugging-tips-and-tricks)
- [Opset specification](#opset-specification)
  - [Examine layer specification](#examine-layer-specification)
- [Single layer test](#single-layer-test)
  - [Create a new file with a test](#create-a-new-file-with-a-test)
- [IE Dialect](#ie-dialect)
  - [IE Operation table gen](#ie-operation-table-gen)
  - [NGraph parser](#ngraph-parser)
  - [IE Output shape resolver](#ie-output-shape-resolver)
  - [Optional: IE Attribute parser](#optional-ie-attribute-parser)
  - [Optional: Canonicalizer](#optional-canonicalizer)
  - [Optional: Transformation pass](#optional-transformation-pass)
- [VPU Dialect](#vpu-dialect)
  - [VPU Operation table gen](#vpu-operation-table-gen)
  - [VPU op tiling](#vpu-op-tiling)
    - [Tiling lit-test](#tiling-lit-test)
    - [Tiling functional tests](#tiling-functional-tests)
  - [VPU Output shape resolver](#vpu-output-shape-resolver)
  - [Optional: Attributes, canonicalization, transformations](#optional-attributes-canonicalization-transformations)
- [IE → VPU lowering](#ie--vpu-lowering)
  - [IE → VPU lowering lit-test](#ie--vpu-lowering-lit-test)
- [You're half way there.](#youre-half-way-there)
- [VPUIP Dialect](#vpuip-dialect)
  - [VPUIP table gen](#vpuip-table-gen)
  - [Redirect interfaces for IE and VPUIP](#redirect-interfaces-for-ie-and-vpuip)
  - [VPUIP verifier](#vpuip-verifier)
  - [Kernel binaries](#kernel-binaries)
  - [Add kernel information](#add-kernel-information)
- [VPU → VPUIP lowering](#vpu--vpuip-lowering)
  - [Special solution for optional inputs](#special-solution-for-optional-inputs)
  - [VPU → VPUIP lowering lit-test](#vpu--vpuip-lowering-lit-test)
# Introduction
This instruction will guide you through steps of adding a new software layer to the MLIR compiler. It has step-by-step plan of actions using `Sigmoid` layer as an example. 
> Be aware, that MLIR compiler is in a rapid development and code snippets might be out of date.

# Debugging tips and tricks
Make sure to take a look at [debugging documentation](guides/how_to_debug.md) to have a common knowledge of technics and tools that will help you investigate problems when developing a layer.

# Opset specification
* [OpenVINO web site](https://docs.openvinotoolkit.org/latest/operations_specifications.html)
* [OpenVINO github](https://github.com/openvinotoolkit/openvino/tree/master/docs/ops)


## Examine layer specification

Let's implement [Sigmoid](https://docs.openvino.ai/2022.3/openvino_docs_ops_activation_Sigmoid_1.html) operation from `OpenVINO opset-1`.

https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/op/sigmoid.hpp
If you found, that ov don't follow the operation specification, you should create a bug ticket.

`Sigmoid-1`:

Inputs:
* `input` is a floating point tensor of shape.

Outputs:
* `output` is a floating point tensor with shape and type matching the input tensor.

> Things to keep in mind:
> * Input count, size and type.
> * Output count, size and type.
> * Attribute types.
> * Are any of the above optional.

# Single layer test

Add OpenVINO single layer test. Copy test suites from the MKLDNN plugin for initial setup.

A simple test will be useful to have for debugging. Run it to see the build/compilation issues.
Make sure to derive `LayerTest` from `VpuOv2LayerTest`.

Useful links:
[How to run tests](../../../guides/how-to-test.md)

## Create a new file with a test
[tests/functional/shared_tests_instances/single_layer_tests/activation.cpp](../../../tests/functional/shared_tests_instances/single_layer_tests/activation.cpp)
```cpp
#include "single_op_tests/ctc_greedy_decoder.hpp"
#include "vpu_ov2_layer_test.hpp"
#include <vector>

using namespace ov::test::utils;
using ov::test::ActivationParamLayerTest;

namespace ov::test {
class ActivationLayerTestCommon : public ActivationLayerTest, virtual public VpuOv2LayerTest {};

class ActivationLayerTest_SW_FP16 : public ActivationLayerTestCommon {};
class ActivationLayerTest_HW_FP16 : public ActivationLayerTestCommon {};

// SW
TESTP(ActivationLayerTest_SW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

// HW
TEST_P(ActivationLayerTest_HW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid, {{1.0f}}},
        ...
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic = {{{{1, 50, 1, 1}}, {}}, {{{1, 128, 1, 1}}, {}}};

auto static_shapes_param_transform =
        [](const std::vector<std::pair<std::vector<ov::Shape>, ov::Shape>>& original_shapes) {
            std::vector<std::pair<std::vector<ov::test::InputShape>, ov::Shape>> new_shapes;
            for (const auto& shape_element : original_shapes) {
                new_shapes.emplace_back(ov::test::static_shapes_to_test_representation(shape_element.first),
                                        shape_element.second);
            }
            return new_shapes;
        };

const auto basicCases =
        ::testing::Combine(::testing::ValuesIn(::combineParams(activationTypes)),  // Activation type and constant
                           ::testing::ValuesIn(netPrecisions),                     // Model type
                           ::testing::ValuesIn(static_shapes_param_transform(
                                   ov::test::utils::combineParams(basic))),  // Input shapes and input const shape
                           ::testing::Values(test_utils::TARGET_DEVICE));    // Target device name

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation, ActivationLayerTest_SW_FP16, basicCases,
                         ActivationLayerTest::getTestCaseName);
```

# IE Dialect
The IE Dialect represents InferenceEngine/nGraph IR in terms of MLIR framework.

It has the following properties:

* Describes network topology without HW details (memory hierarchy, memory allocation, scheduling).
* Represents the latest nGraph opset and in addition some portion of legacy IE opset (for convenience).
* Works with MLIR Tensor Types as atomic Values (no memory effects), all operations are pure.
* Performs high level transformations/optimizations, that doesn't need low level details (memory buffers, scheduling).

Documentation

* [chapters/generated/dialect/_IE.md](chapters/generated/dialect/_IE.md)

## IE Operation table gen
Let's create a table-gen representation of our layer.

* let summary – one line description of op
* let arguments – input parameters for the layer. Possible types can be found here: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td
* let results – outputs of the operation

[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/ops.td#L2018](../tblgen/vpux/compiler/dialect/IE/ops.td#L2018)
```swift
//
// SigmoidOp
//

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}
```

## NGraph parser

Define parseNode function, that will transform ov operation to MLIR representation.

[src/vpux_compiler/include/vpux/compiler/frontend/IE.hpp#L123](../include/vpux/compiler/frontend/IE.hpp#L123)
```cpp
class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ov::Model> netGraph, bool sharedConstants, Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    // Declare parser for ov operation
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sigmoid>& origNode);
}
```
Check input tensors and parse ov operation.

[src/vpux_compiler/src/frontend/IE.cpp#L1722](../src/frontend/IE.cpp#L1722)
```cpp
void NGraphImporter::parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::Sigmoid>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::Sigmoid>::value,
                  "opset operation mismatch");

    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 1, "nGraph Sigmoid node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::SigmoidOp>(createLocation(origNode), inputs[0]);

    addOutputs(origNode, op);
}
```
Add map entry for operation dispatcher.

[src/vpux_compiler/src/frontend/IE.cpp#L144](../src/frontend/IE.cpp#L144)
```cpp
NGraphImporter::Callback NGraphImporter::getParser(const std::shared_ptr<ov::Node>& op) {
    using DispatchMap = std::map<ov::NodeTypeInfo, Callback>;

#define MAP_ENTRY(_NodeType_) \
    { _NodeType_::get_type_info_static(), &NGraphImporter::parseDispatch<_NodeType_> }

    static const DispatchMap dispatchMap{
            {ov::op::v0::Parameter::get_type_info_static(), &NGraphImporter::parseEmpty},
            {ov::op::v0::Result::get_type_info_static(), &NGraphImporter::parseEmpty},

            MAP_ENTRY(ov::opset1::Sigmoid),
    };

#undef MAP_ENTRY

    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    return (dispatchIt != dispatchMap.end()) ? dispatchIt->second : nullptr;
}
```

## IE Output shape resolver
Create a new file that defines the `vpux::IE::<OpName>::inferReturnTypeComponents` function.
Given input tensors and layer parameters, this function computes output shapes and types of the operation.
[(new) src/vpux_compiler/src/dialect/IE/IR/ops/sigmoid.cpp](../src/dialect/IE/IR/ops/sigmoid.cpp)
```cpp
mlir::LogicalResult vpux::IE::SigmoidOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SigmoidOpAdaptor sigmoid(operands, attrs, prop);
    if (mlir::failed(sigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(sigmoid.getInput().getType());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}
```

## Optional: IE Attribute parser
For the operations with sophisticated parameters (eg parameters cannot be expressed with numbers, enum), custom attribute should be implemented. This attribute is not related to the example above.
[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/attributes.td#L124](../tblgen/vpux/compiler/dialect/IE/attributes.td#L124)
```swift
//
// RoundingType
//

def IE_RoundingType :
        IE_I64EnumAttr<
            "RoundingType",
            "Rounding type that operations support",
            [
                I64EnumAttrCase<"FLOOR">,
                I64EnumAttrCase<"CEIL">,
            ]
        > {
}

def IE_RoundingTypeAttr : IE_EnumAttr<IE_RoundingType, "rounding_type">;

```
Additional helper function should be used for parsing the attribute.
[src/vpux_compiler/src/frontend/IE.cpp#L164](../src/frontend/IE.cpp#L164)
```cpp
private:
    IE::RoundingTypeAttr importRoundingType(ov::op::RoundingType roundingType);
```
[src/vpux_compiler/src/frontend/IE.cpp#L1333](../src/frontend/IE.cpp#L1333)
```cpp
IE::RoundingTypeAttr NGraphImporter::importRoundingType(ov::op::RoundingType roundingType) {
    switch (roundingType) {
    case ov::op::RoundingType::FLOOR:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::FLOOR);
    case ov::op::RoundingType::CEIL:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}
```

## Optional: Canonicalizer

IE Dialect operation can contain canonicalization pattern, which simplifies IR (fusing, using more concrete operations, converting constant operands to attribute).
Such manipulation should be done on IE Dialect level, not ngraph parser, because FrontEnd just performs 1-to-1 mapping without any transformation/adaptation logic, and with such separation we have simple frontend and Canonicalization pass, which we can cover with tests.

Most used case is converting inputs (e.g. parameters from weights) into attributes. In this case we will simplify our graph (less edges between constant and layer) and simplify approach how to work with attributes (because in case of working / manipulating with inputs, we need first check, that it's constant, then transform it, etc.)

`Swish` operation canonicalizer example
[src/vpux_compiler/src/dialect/IE/IR/ops/swish.cpp#L41](../src/dialect/IE/IR/ops/swish.cpp#L41)
```cpp
//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SwishOp> {
public:
    using mlir::OpRewritePattern<IE::SwishOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const {
    // Check if Input was already converted to Attribute
    auto beta = swishOp.beta();
    if (beta == nullptr) {
        return mlir::failure();  // mlir::failure() means that pass was not applied
    }

    // Check for Input to be a constant
    auto betaOp = swishOp.beta().getDefiningOp<Const::DeclareOp>();
    if (betaOp == nullptr) {
        return mlir::failure();
    }

    // Check for constant to have "one value"
    const auto betaContent = betaOp.content();
    if (!betaContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::SwishOp>(swishOp, swishOp.getType(), swishOp.input(), nullptr,
                                             rewriter.getF64FloatAttr(betaContent.getSplatValue<float>()));

    return mlir::success();
}

}  // namespace

void vpux::IE::SwishOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
```
Add hasCanonicalizer variable to table gen defenition
[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/ops.td#1531](../tblgen/vpux/compiler/dialect/IE/ops.td#1531)
```swift
def IE_SwishOp :
    ...
    let hasCanonicalizer = 1;
}
```
Some notes about `mlir::failure();`. It doesn't mean, that pass failed. It just mean, that this pass cannot be applied and should be skipped. In example above, this line mean we already converted input into attr, and don't need to do it again. Since canonicalizer pass can be executed few time, we can end-up in endless loop trying to apply this optimization, if we don't do such check.

## Optional: Transformation pass
Canonicalizer, as described, is simple version of transformation. We can do simple fusing, parameters manipulation, but, in general, we will stay with the same operation as before, but in a canonical state.

If we need to do something more complicated, we should be using a pass instead.

Documentation
* https://mlir.llvm.org/docs/DialectConversion

Let's take a look at the example of supporting 1D Convolution. We have Convolution2D already supported. By converting 1D Convolution to 2D variant, we can support Convolution1D operation without an actual kernel implementation.

[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/passes.td#L287](../tblgen/vpux/compiler/dialect/IE/passes.td#L287)
```swift
//
// ConvertConv1DToConv2D
//

def ConvertConv1DToConv2D : PassBase<"convert-conv1d-to-conv2d", "vpux::FunctionPass"> {
    let summary = "Convert Convolution1D and GroupConvolution1D to its 2D variance";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Extends input, filter and output tensors with height = 1.
        [N, C, W] -> [N, C, 1, W]
        strides:    {2} -> strides:    {1, 2}
        pads_begin: {2} -> pads_begin: {0, 2}
        pads_end:   {2} -> pads_end:   {0, 2}
        dilations:  {2} -> dilations:  {1, 2}
    }];

    let constructor = "vpux::IE::createConvertConv1DToConv2DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}
```
Declare a function, that will instantiate custom pass.

[src/vpux_compiler/include/vpux/compiler/dialect/IE/transforms/passes.hpp](../include/vpux/compiler/dialect/IE/transforms/passes.hpp)
```cpp
// Adjust IE Dialect IR for VPU target.
...
std::unique_ptr<mlir::Pass> createConvertMultiplyToLegacyPowerPass(Logger log = Logger::global());
...
```

Create pass implementation file. Define rewriter pass and derive from `mlir::OpRewritePattern`.
There is also more sophisticated `mlir::OpConversionPattern` you might use. https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
//
// ConvolutionExpansion
//

class ConvolutionExpansion final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionExpansion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionExpansion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
```
Write main pass logic that `matchesAndRewrites` desired operations

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
mlir::LogicalResult ConvolutionExpansion::matchAndRewrite(IE::ConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Convolution Operation '{0}'", origOp->getLoc());

    // Transform inputs and attributes
    const auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp.input());
    const auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.filter());
    const auto newBias = extendTensor(rewriter, origOp->getLoc(), origOp.bias());

    const auto newStrides = append(getContext(), origOp.strides(), 1);
    const auto newPadsBegin = append(getContext(), origOp.pads_begin(), 0);
    const auto newPadsEnd = append(getContext(), origOp.pads_end(), 0);
    const auto newDilations = append(getContext(), origOp.dilations(), 1);

    // Create new operation with transformed parameters
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                        newPadsBegin, newPadsEnd, newDilations, origOp.post_opAttr());

    const auto outputShape = mlir::cast<mlir::ShapedType>(origOp.output().getType()).getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Replace old IE::ConvolutionOp with a new IE::ConvolutionOp + IE::ReshapeOp
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::Convolution' (2D)");

    return mlir::success();
}
```
After defining a match and rewite pattern, create `safeRunOnFunc()` function.
1. It contains a list of operations that should be `legalized`. In our case its a `Convolution1D` operation.
2. `Convolution1D` has 3D input tensor and is considered illegal. `isLegalConvOp` will return `true` if given Convolution has `input != 3D`, thus should not be converted by our pass.
3. Create `ConversionTarget` and list all the operations involved in a transformation.
4. Add convertion patterns that will try to legalize all `DynamicallyLegalOps`.
5. Use `applyPartialConversion` function to run the pass. More conversion modes could be found in the [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) documentation.

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
//
// ConvertConv1DToConv2DPass
//

class ConvertConv1DToConv2DPass final : public IE::ConvertConv1DToConv2DBase<ConvertConv1DToConv2DPass> {
public:
    explicit ConvertConv1DToConv2DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertConv1DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    // Illegal ops will be converted (legalized)
    const auto isLegalConvOp = [&](IE::ConvolutionOp conv) {
        const auto inputShape = mlir::cast<mlir::ShapedType>(conv.input().getType()).getShape();
        return inputShape.size() != 3;
    };

    mlir::ConversionTarget target(ctx);
    // DynamicallyLegalOp is illegal op that could be legalized
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalConvOp);
    // Add legal ops that also are used in a transformation
    // Usually it will be IE::ReshapeOp, IE::ConvertOp or similar
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionExpansion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConv1DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConv1DToConv2DPass(Logger log) {
    return std::make_unique<ConvertConv1DToConv2DPass>(log);
}
```
Add pass to the pipeline. Most of the transormations should be added to `buildAdjustForVPUPipeline` because they are specific to VPU platform.

[src/vpux_compiler/src/dialect/IE/transforms/pipelines.cpp](../src/dialect/IE/transforms/pipelines.cpp)
```cppvoid vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log) {
    ...
    pm.addPass(IE::createConvertConv1DToConv2DPass(log));
}

```

# VPU Dialect
The VPU Dialect represents an extension over the IE dialect that brings hardware-specific information to the IR.

Documentation

* [chapters/generated/dialect/_VPU.md](chapters/generated/dialect/_VPU.md)

## VPU Operation table gen
Let's create a table-gen representation of our layer.

* let summary – one line description of op
* let arguments – input parameters for the layer. Possible types can be found here: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td
* let results – outputs of the operation

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPU/ops.td#L2132](../tblgen/vpux/compiler/dialect/VPU/ops.td#L2132)
```swift
//
// Sigmoid
//

def VPU_SigmoidOp :
        VPU_LayerOp<
            "Sigmoid",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Sigmoid VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}
```

## VPU op tiling

Software ops need to have their data fit into NNCMX in order to execute. Therefore, they should be tiled into multiple smaller operations if they do not fit. For this to happen, every operation needs to have:

- the `VPU::TilingBuilderOpInterface` interface attached or inerhited;
- an implementation for the `VPU::TilingBuilderOpInterface::backInferTileInfo` method, which returns the information on the tiles of the input operands when given an output tile (i.e. a smaller part of the output);
- an implementation for the `VPU::TilingBuilderOpInterface::getTilingStrategy` methods, which returns the optimal output tiling scheme
  fot this particular operation;
- the `VPU::TilingInfoOpInterface` interface attached or inherited;
- an implementation for the `VPU::TilingInfoOpInterface::isSupportedTiling` method, which returns whether the data used by the operation for a given output tile fits into memory; it generally makes use of the `backInferTileInfo` mentioned above to take the inferred input tiles into account.

The simplest case of enabling tiling for a software operation is when the operation is element-wise: one element in the output corresponds to one element in the input. In such cases, it is enough to have the operation inherit the two following interfaces in [src/vpux_compiler/tblgen/vpux/compiler/dialect/VPU/ops.td](../tblgen/vpux/compiler/dialect/VPU/ops.td):

```
VPU_TilingBuilderOpInterface,
VPU_EltwiseOp
```

The `VPU::EltwiseOp` interface comes with an implementation for the `backInferTileInfo` method that returns the input tile(s) equal to the output tile. Then, `VPU::TilingInfoOpInterface` can be attached to the operation in [src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp). Example:

```cpp
VPU::SigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SigmoidOp>>(*ctx);
```

`SwLayerTilingInfoOpModel` is an implementation of `VPU::TilingInfoOpInterface` for software layer tiling that contains dispatch methods for computing the NNCMX usage based on the operation type. For element-wise operations, a generic method adds up the size of the output and input tiles.

In case your operation is more complex, it might be necessary to provide a dedicated implementation for the `backInferTileInfo` method and/or the dispatch method used by `isSupportedTiling`. See `VPU.MemPermute` as an example.

### Tiling lit-test

To ensure that tiling is functional for your operation, a lit-test should be created. `PrefetchTiling` is recommended and should be checked with two steps:
- Check if the op is assigned with the desired tiling strategy: [tests/lit/NPU/dialect/VPU/passes/tiling_strategy_assignment_prefetch.mlir](../../../tests/lit/NPU/dialect/VPU/passes/tiling_strategy_assignment_prefetch.mlir)
- Check if the op is tiled correctly with assigned strategy: [tests/lit/NPU/dialect/VPU/passes/apply_tiling.mlir](../../../tests/lit/NPU/dialect/VPU/passes/apply_tiling.mlir)

### Tiling functional tests

To verify at runtime that the tiling logic is applied, a functional test case with large input values should be added. Example from the [Activation group](../../../tests/functional/shared_tests_instances/single_layer_tests/activation.cpp):

```cpp
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicTiling = {{{1, 8, 80, 1280}, {{}}}};

```

For groups, such as Eltwise, Activation, Comparison etc. it is not mandatory to have functional test cases on the main developing branch for all of the operators that are enabled, as the tiling logic is the same and also to avoid overloading the CI. They should be tested locally beforehand.

An example would be the `activationTypesTiling` variable, which does not contain all of the Activation operators:

```
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesTiling = {
        {Sigmoid, {{1.0f}}}, {Elu, {{1.0f}}},        {Sqrt, {{1.0f}}}, {Exp, {{1.0f}}},  {Clamp, {{-1.0f, 1.0f}}},
        {Tanh, {{1.0f}}},    {LeakyRelu, {{0.01f}}}, {Log, {{1.0f}}},  {Relu, {{1.0f}}}, {Negative, {{0.01f}}}};
```

 In case of an operator that is standalone, a test case where the tiling logic is tested should always be present. Please see Interpolate [example](../../../tests/functional/shared_tests_instances/single_layer_tests/interpolate.cpp).

## VPU Output shape resolver
Create a new file that defines the `vpux::VPU::<OpName>::inferReturnTypes` function.
Compared to the IE dialect, this function will return the output types of the operation, not their shape components. This allows the operations in the dialect to work with any type, without begin restricted to `mlir::ShapedType`.

It is recommended to opt for `vpux::NDTypeInterface` while working with tensor types in this dialect, since this interface is compatible with all MLIR and custom types.

[(new) src/vpux_compiler/src/dialect/VPU/IR/ops/sigmoid.cpp](../src/dialect/VPU/IR/ops/sigmoid.cpp)
```cpp
mlir::LogicalResult vpux::VPU::SigmoidOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SigmoidOpAdaptor sigmoid(operands, attrs, prop);
    if (mlir::failed(sigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = sigmoid.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
```

## Optional: Attributes, canonicalization, transformations

The same information from the IE dialect with regards to attributes, canonicalization and transformations applies to the VPU dialect, with some mentions:

- canonicalization logic to convert inputs (e.g. parameters from weights) into attributes shouldn't generally be necessary if it is present in IE dialect;
- if operands are converted to attributes in IE dialect, there should be no need to add the operands in VPU dialect as well - the attributes should suffice; the operands might be included in cases where the conversion from operands to attributes in IE is done conditionally.

## IE → VPU lowering
Now that the IE and VPU definitions of the operation are created, the logic that performs the IE->VPU lowering for it should follow.

Generally, it will be enough to add the lowering logic in [convert_layers_to_VPU.td](../tblgen/vpux/compiler/conversion/rewriters/convert_layers_to_VPU.td):

```swift
//
// IE.Sigmoid -> VPU.Sigmoid
//

def createSigmoidOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SigmoidOp>($_loc, $0)
        }]>;

def RewriteSigmoid :
        Pat<
            (IE_SigmoidOp $input),
            (createSigmoidOp $input)
        >;
```

This tablegen declaration will automatically generate the C++ code that does the lowering. Sometimes, using the tablegen is not possible since the generated code cannot cover the operands/results properly. Example of relevant cases:
- operation has a variadic number of results;
- operation has multiple results and at least one optional/variadic operand.

For such cases, it will be necessary to manually create the lowering logic in [convert_layers_to_VPU.cpp](../src/conversion/passes/IE2VPU/convert_layers_to_VPU.cpp). Although this is not applicable for `CTCGreedyDecoder` used as example, its manual lowering logic would have looked like:

```cpp
//
// CTCGreedyDecoderRewrite
//

class CTCGreedyDecoderRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderOp> {
public:
    CTCGreedyDecoderRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CTCGreedyDecoderRewrite::matchAndRewrite(IE::CTCGreedyDecoderOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoder Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderOp>(origOp, origOp.input(), origOp.sequenceLengths(),
                                                         origOp.mergeRepeatedAttr());

    return mlir::success();
}
```

The rewritter would also have to be registered:

```cpp
void ConvertLayers2VPUPass::safeRunOnFunc() {
    // ...
    mlir::RewritePatternSet patterns(&ctx);
    // ...
    patterns.add<CTCGreedyDecoderRewrite>(&ctx, _log);
    // ...
```

### IE → VPU lowering lit-test

The lowering logic should also be tested. For this, create a dedicated lit-test in [convert_layers_to_VPU.mlir](../../../tests/lit/NPU/conversion/passes/IE2VPU/convert_layers_to_VPU.mlir) containing the IE operation as input and checks for the resulting VPU operation. If needed, other operations such as constants can be included. Example for `VPU.CTCGreedyDecoder` operation:

```cpp
// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated}
    // CHECK-SAME:    : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<8x20x1x1xf16>
}
```

## You're half way there.
You should be able to compile code now. Run single layer test and look for `Unable to legalize VPU::OperationName` message. That means that MLIR compiler was not able to convert VPU::OperationName to VPUIP::OperationName. This will be the next step.

# VPUIP Dialect

In VPUIP dialect all NPU3720 software layers are represented via `VPUIP::SwKernelOp` operation.

This dialect no longer works with tensor data. Instead, buffers are utilized by making use of `MemRefType`.

Documentation
* VPUIP dialect: [chapters/generated/dialect/_VPUIP.md](chapters/generated/dialect/_VPUIP.md#L1)
* Passes: [chapters/generated/dialect/VPUIP/_passes.md](chapters/generated/dialect/VPUIP/_passes.md#L1)
* Op interfaces: [chapters/generated/dialect/VPUIP/_ops_interfaces.md](chapters/generated/dialect/VPUIP/_ops_interfaces.md#L1)
* About assembly format: [https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format](https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format)

## VPUIP table gen

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPUIP/ops.td#L2078](../tblgen/vpux/compiler/dialect/VPUIP/ops.td#L2419)
```swift
//
// VPUIP_SwKernelOp
//

def VPUIP_SwKernelOp :
        VPUIP_TaskOp<1, "SW.Kernel",
            [
                MultiViewOpInterface,
                IsolatedFromAbove,
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Software Layer Task";

    let description = [{
        This operation defines Activation shave task.
        There are two different modes or handling inputs with dynamic shapes
        - In the first mode, dynamic inputs are accepted as BoundedBuffers, which combine data and dynamic shape into one type.
        dynamicInputShapes and dynamicInputShapesMap are not used in that case:
                inputs[3]: [BoundedBuffer, MemRef, BoundedBuffer]
                dynamicInputShapes[]: []
                dynamicInputShapesMap[]: nullptr
        - In the second mode all BoundedBuffer's are unrolled by ungroup-bounded-buffers pass: E#111348.
        Separate handling of dynamic data and dynamic shape is needed because feasible allocation doesn't support multiple root buffers per input
                inputs[3]: [MemRef, MemRef, MemRef]
                dynamicInputShapes[2]: [MemRef, MemRef]
                dynamicInputShapesMap[3]: [0, -1, 1]
        For outputs, the same applies.
    }];

    let arguments = (ins
        SymbolRefAttr:$kernelFunction,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$inputs,
        Variadic<MemRefOf<[SI32]>>:$dynamicInputShapes,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicInputShapesMap,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$output_buffs,
        Variadic<MemRefOf<[SI32]>>:$dynamicOutputShapeBuffs,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicOutputShapesMap,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_data,
        OptionalAttr<IntAttr>:$tileIndex,
        OptionalAttr<I64ArrayOfArraysAttr>:$strides,
        OptionalAttr<VPUIP_SwProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$results,
        Variadic<MemRefOf<[SI32]>>:$dynamicOutputShapes,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_output
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let builders = [
        ...
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_ACT;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                    mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);

        static vpux::VPUIP::KernelInfo getKernelInfo(mlir::Operation* origOp);
        static vpux::VPUIP::KernelInfo getDummyKernelInfo();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];

    let hasVerifier = 1;
}
```

## VPUIP verifier
Verifiers are used to validate state of the operation. It is common to check input size, layout and strides for correctness. Add checks for kernel limitations if present.

Add verifier to the VPUIP table gen.

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPUIP/ops.td](../tblgen/vpux/compiler/dialect/VPUIP/ops.td)
```swift
let hasVerifier = 1;
```
Implement verify function.
[src/vpux_compiler/src/dialect/VPUIP/IR/ops/dma.cpp](../src/dialect/VPUIP/IR/ops/dma.cpp)
```cpp
mlir::LogicalResult SwKernelOp::verify() {
    const auto op = getOperation();
    if (VPUIP::isCacheHandlingOp(*this)) {
        if (!op->getOperands().empty()) {
            return errorAt(op, "SW Kernel Cache Op should have no operands");
        }
        if (!op->getResults().empty()) {
            return errorAt(op, "SW Kernel Cache Op should have no results");
        }
        auto kernelFunc =
                op->getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::func::FuncOp>(getKernelFunctionAttr());
        if (kernelFunc.getFunctionType().getNumInputs() != 0) {
            return errorAt(op, "SW Kernel Cache Op func should have no inputs");
        }

        if (kernelFunc.getFunctionType().getNumResults() != 0) {
            return errorAt(op, "SW Kernel Cache Op func should have no results");
        }
        return mlir::success();
    }
    ...
    return mlir::success();
}
```

## Kernel binaries

[act_shave_bin](../../../sw_runtime_kernels/kernels/prebuild/act_shave_bin) folder should contain the following data:

- sk.`<entry point>`.`<platform>`.data
- sk.`<entry point>`.`<platform>`.text

If not please follow this instruction: [How to create act-shave kernel](../../../sw_runtime_kernels/README.md)

## Add kernel information

To serialize the kernel, you need to provide additional information about the arguments of the kernel, the name of entry point and source file. This information is stored in the structure:

[src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp](../include/vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp)

```cpp
struct KernelInfo final {
    SmallVector<mlir::Attribute> args;
    SmallString entryName;
    SmallString sourceFileName;
};
```

Provide the necessary information:
[src/vpux_compiler/src/dialect/VPUIP/IR/ops/sw_kernel.cpp](../src/dialect/VPUIP/IR/ops/sw_kernel.cpp)

```cpp
VPUIP::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    return llvm::TypeSwitch<mlir::Operation*, VPUIP::KernelInfo>(origOp)
            .Case<VPU::SigmoidOp>([&](VPU::SigmoidOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sigmoid"}};
            })
            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}
```

## VPU → VPUIP lowering
Convert previous representation of a layer in VPU dialect down to the VPUIP dialect.

[src/vpux_compiler/src/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.cpp](../src/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.cpp)
```cpp
void vpux::registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        ...
        VPU::SigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SigmoidOp>>(*ctx);
        ...
}
```

### Special solution for optional inputs
Some operations like FullyConnected/EmbeddingBagPackedSum have optional input, adding a delimeter to help check the count of MemRefData(operands are represented by MemRefData) in specifc case.

##### Adding delimiter attr at VPUIP layer
[src/vpux_compiler/src/dialect/VPUIP/IR/ops/sw_kernel.cpp](../src/vpux_compiler/src/dialect/VPUIP/IR/ops/sw_kernel.cpp)
```cpp
    const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr}, {"fully_connected"}};
```

##### Param struct of optional inputs 
```cpp
struct LayerData {
    struct MemRefData tensors[MAX_TENSOR_COUNT]; // MAX_TENSOR_COUNT==N
    int64_t memRefDelimiter;
}
```

#### Example
```cpp
// actual data:
[in1, optional<in2>, nullptr, nullptr, delimiter, attr1, attr2]
// optional<in3>,..., optional<in(MAX_TENSOR_COUNT-1)> are absent

//init inputs in kernel
auto memrefCount = countMemrefs(layerData, MAX_TENSOR_COUNT);
in1 = nullptr;
in2 = nullptr;
...
inN = nullptr;
For memrefIdx from 0 to memrefCount - 1:
    in[memrefIdx] = layerData[memrefIdx]

```

### VPU → VPUIP lowering lit-test
Similar to IE->VPU, the lowering logic will be tested by creating a lit-test in [bufferize_sw_ops_to_VPUIP_sw_kernel_37XX+.mlir](../../../tests/lit/NPU/conversion/passes/VPU2VPUIP/bufferize_sw_ops_to_VPUIP_sw_kernel_37XX+.mlir). If there are instances where the lowering logic behaves differently for various configurations of the operation, please make sure to cover them all with different tests.

```cpp
// CHECK-LABEL:  func.func @ReduceSumSWLayer
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x7x2x3xf16, #NHWC>)
func.func @ReduceSumSWLayer(%input: tensor<1x7x2x3xf16, {order = #NHWC}>) -> tensor<1x1x2x3xf16, {order = #NHWC}> {
    %output = VPU.ReduceSum(%input) {axes_value = [1], keep_dims} : tensor<1x7x2x3xf16, {order = #NHWC}> -> tensor<1x1x2x3xf16, {order = #NHWC}>
    return %output : tensor<1x1x2x3xf16, {order = #NHWC}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x7x2x3xf16, #NHWC>) outputs([[ALLOC]] : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs([[COPY0]] as {{[^:]+}}: memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC0]] as {{[^:]+}}: memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1, 1, [0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x2x3xf16, #NHWC>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x1x2x3xf16, #NHWC>) -> memref<1x1x2x3xf16, #NHWC>
    // CHECK: return [[COPY1]] : memref<1x1x2x3xf16, #NHWC>
}
```

When the operation is lowered to VPUIP, the SW Kernel task will work with buffer data, or `memrefs`, instead of `tensors`.
