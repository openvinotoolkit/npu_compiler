//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Parser/Parser.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using namespace vpux;

namespace {
class MLIR_ContentSetupTest : public MLIR_UnitBase {
public:
    MLIR_ContentSetupTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    mlir::MLIRContext ctx;

    Const::ContentSetup getContentSetup(ArrayRef<int64_t> shape, mlir::Type type) {
        const auto baseType = mlir::RankedTensorType::get(shape, type);
        return Const::ContentSetup(baseType);
    }

    template <typename T>
    Const::ContentAttr getContentAttr(ArrayRef<int64_t> shape, mlir::Type elemType, ArrayRef<T> data) {
        auto rankedType = mlir::RankedTensorType::get(shape, elemType);
        auto baseAttr = Const::createConstContent(rankedType, data);
        return Const::ContentAttr::get(baseAttr);
    }

    void checkSubViewAttr(Const::TransformAttrInterface actualTransformation, ArrayRef<int64_t> expectedOffset,
                          ArrayRef<int64_t> expectedShape) {
        auto actualSubView = mlir::dyn_cast<Const::SubViewAttr>(actualTransformation);
        ASSERT_TRUE(actualSubView != nullptr);

        auto actualOffset = parseIntArrayAttr<int64_t>(actualSubView.getOffset());
        auto actualShape = parseIntArrayAttr<int64_t>(actualSubView.getShape());

        EXPECT_THAT(actualOffset, testing::ElementsAreArray(expectedOffset));
        EXPECT_THAT(actualShape, testing::ElementsAreArray(expectedShape));
    }

    void checkPadWithZeroAttr(Const::TransformAttrInterface actualTransformation, ArrayRef<int64_t> expectedPadBefore,
                              ArrayRef<int64_t> expectedPadAfter) {
        auto actualPadWithZero = mlir::dyn_cast<Const::PadWithZeroAttr>(actualTransformation);
        ASSERT_TRUE(actualPadWithZero != nullptr);

        auto actualPadBefore = parseIntArrayAttr<int64_t>(actualPadWithZero.getPadBefore());
        auto actualPadAfter = parseIntArrayAttr<int64_t>(actualPadWithZero.getPadAfter());

        EXPECT_THAT(actualPadBefore, testing::ElementsAreArray(expectedPadBefore));
        EXPECT_THAT(actualPadAfter, testing::ElementsAreArray(expectedPadAfter));
    }

    void checkChangeShapeAndElemTypeAttr(Const::TransformAttrInterface actualTransformation,
                                         ArrayRef<int64_t> expectedShape, mlir::Type expectedElemType) {
        auto actualChangeShapeAndElemTypeAttr = mlir::dyn_cast<Const::ChangeShapeAndElemTypeAttr>(actualTransformation);
        ASSERT_TRUE(actualChangeShapeAndElemTypeAttr != nullptr);

        auto actualShape = parseIntArrayAttr<int64_t>(actualChangeShapeAndElemTypeAttr.getShape());
        auto actualElemType = actualChangeShapeAndElemTypeAttr.getElemType();

        EXPECT_THAT(actualShape, testing::ElementsAreArray(expectedShape));
        EXPECT_THAT(actualElemType, expectedElemType);
    }

    void checkReshapeAttr(Const::TransformAttrInterface actualTransformation, ArrayRef<int64_t> expectedShape) {
        auto actualReshape = mlir::dyn_cast<Const::ReshapeAttr>(actualTransformation);
        ASSERT_TRUE(actualReshape != nullptr);

        auto actualShape = parseIntArrayAttr<int64_t>(actualReshape.getShape());

        EXPECT_THAT(actualShape, testing::ElementsAreArray(expectedShape));
    }

    void checkDequantizeAttr(Const::TransformAttrInterface actualTransformation) {
        auto actualDequant = mlir::dyn_cast<Const::DequantizeAttr>(actualTransformation);
        EXPECT_TRUE(actualDequant != nullptr);
    }

    void checkAddAttr(Const::TransformAttrInterface actualTransformation, double expectedValue) {
        const auto getValue = [](Const::AddAttr attr) {
            return attr.getBias().getValueAsDouble();
        };
        checkSingleValueAttr<Const::AddAttr, double>(actualTransformation, expectedValue, getValue);
    }

    void checkRescaleAttr(Const::TransformAttrInterface actualTransformation, double expectedValue) {
        const auto getValue = [](Const::RescaleAttr attr) {
            return attr.getScale().getValueAsDouble();
        };
        checkSingleValueAttr<Const::RescaleAttr, double>(actualTransformation, expectedValue, getValue);
    }

    void checkReorderAttr(Const::TransformAttrInterface actualTransformation, DimsOrder expectedValue) {
        const auto getValue = [](Const::ReorderAttr attr) {
            return DimsOrder::fromAffineMap(attr.getOrder().getValue());
        };
        checkSingleValueAttr<Const::ReorderAttr, DimsOrder>(actualTransformation, expectedValue, getValue);
    }

    void checkTransposeAttr(Const::TransformAttrInterface actualTransformation, DimsOrder expectedValue) {
        const auto getValue = [](Const::TransposeAttr attr) {
            return DimsOrder::fromAffineMap(attr.getOrder().getValue());
        };
        checkSingleValueAttr<Const::TransposeAttr, DimsOrder>(actualTransformation, expectedValue, getValue);
    }

    void checkMemPermuteAttr(Const::TransformAttrInterface actualTransformation, DimsOrder expectedOrder,
                             DimsOrder expectedMemPerm) {
        auto memPermAttr = mlir::dyn_cast<Const::MemPermuteAttr>(actualTransformation);
        ASSERT_TRUE(memPermAttr != nullptr);

        auto actualOrder = DimsOrder::fromAffineMap(memPermAttr.getDstOrder().getValue());
        EXPECT_THAT(actualOrder, expectedOrder);

        auto actualMemPerm = DimsOrder::fromAffineMap(memPermAttr.getMemPerm().getValue());
        EXPECT_THAT(actualMemPerm, expectedMemPerm);
    }

    void checkCastElemTypeAttr(Const::TransformAttrInterface actualTransformation, mlir::Type expectedValue) {
        const auto getValue = [](Const::CastElemTypeAttr attr) {
            return attr.getElemType();
        };
        checkSingleValueAttr<Const::CastElemTypeAttr, mlir::Type>(actualTransformation, expectedValue, getValue);
    }

    void checkConvertElemTypeAttr(Const::TransformAttrInterface actualTransformation, mlir::Type expectedValue) {
        const auto getValue = [](Const::ConvertElemTypeAttr attr) {
            return attr.getElemType();
        };
        checkSingleValueAttr<Const::ConvertElemTypeAttr, mlir::Type>(actualTransformation, expectedValue, getValue);
    }

    void checkRelocateWeightsTableAttr(Const::TransformAttrInterface actualTransformation,
                                       Const::RelocateWeightsTableAttr expectedAttr) {
        auto actualAttr = mlir::dyn_cast<Const::RelocateWeightsTableAttr>(actualTransformation);
        ASSERT_TRUE(actualAttr != nullptr);
        ASSERT_TRUE(actualAttr == expectedAttr);
    }

    void checkFuseAttr(Const::TransformAttrInterface actualTransformation) {
        auto actualFuse = mlir::dyn_cast<Const::FuseWeightsAttr>(actualTransformation);
        ASSERT_TRUE(actualFuse != nullptr);
    }

    void checkFuseAttrAfterSubView(Const::TransformAttrInterface actualTransformation, bool wtPresent, bool wtSliced,
                                   ArrayRef<int64_t> wtExpectedOffset, ArrayRef<int64_t> wtExpectedShape, bool wtFlat,
                                   ArrayRef<int64_t> wtReshape, ArrayRef<int64_t> wtFlatOffset,
                                   ArrayRef<int64_t> wtFlatShape, bool weightsPresent, bool weightsSliced,
                                   ArrayRef<int64_t> weightsExpectedOffset, ArrayRef<int64_t> weightsExpectedShape,
                                   bool weightsFlat, ArrayRef<int64_t> weightsReshape,
                                   ArrayRef<int64_t> weightsFlatOffset, ArrayRef<int64_t> weightsFlatShape) {
        auto fuse = mlir::dyn_cast<Const::FuseWeightsAttr>(actualTransformation);
        ASSERT_TRUE(fuse != nullptr);

        auto verifyConstant = [this](Const::ContentAttr& constant, bool isSliced, ArrayRef<int64_t> expectedOffset,
                                     ArrayRef<int64_t> expectedShape, bool isFlat, ArrayRef<int64_t> reshape,
                                     ArrayRef<int64_t> flatOffset, ArrayRef<int64_t> flatShape) -> void {
            auto transformations = constant.getTransformations();
            if (isSliced) {
                if (isFlat) {
                    EXPECT_EQ(transformations.size(), 3);
                    checkSubViewAttr(transformations[0], expectedOffset, expectedShape);
                    checkReshapeAttr(transformations[1], reshape);
                    checkSubViewAttr(transformations[2], flatOffset, flatShape);
                } else {
                    EXPECT_EQ(transformations.size(), 1);
                    checkSubViewAttr(transformations[0], expectedOffset, expectedShape);
                }
            } else {
                if (isFlat) {
                    EXPECT_EQ(transformations.size(), 2);
                    checkReshapeAttr(transformations[0], reshape);
                    checkSubViewAttr(transformations[1], flatOffset, flatShape);
                } else {
                    EXPECT_EQ(transformations.size(), 0);
                }
            }
        };

        auto wt = fuse.getWeightsTable();
        if (wtPresent) {
            ASSERT_TRUE(wt != nullptr);
            verifyConstant(wt, wtSliced, wtExpectedOffset, wtExpectedShape, wtFlat, wtReshape, wtFlatOffset,
                           wtFlatShape);
        }
        auto weights = fuse.getWeights();
        if (weightsPresent) {
            ASSERT_TRUE(weights != nullptr);
            verifyConstant(weights, weightsSliced, weightsExpectedOffset, weightsExpectedShape, weightsFlat,
                           weightsReshape, weightsFlatOffset, weightsFlatShape);
        }
    }

private:
    template <typename AttrType, typename ValType>
    void checkSingleValueAttr(Const::TransformAttrInterface actualTransformation, ValType expectedValue,
                              FuncRef<ValType(AttrType)> getValue) {
        auto actualAttr = mlir::dyn_cast<AttrType>(actualTransformation);
        ASSERT_TRUE(actualAttr != nullptr);

        auto actualValue = getValue(actualAttr);
        EXPECT_THAT(actualValue, expectedValue);
    }
};

//
// TEST_SUITE: SwapReshapeAndSubView
//

struct ReshapeAndSubViewInputParams {
    SmallVector<int64_t> reshape;
    SmallVector<int64_t> subViewOffset;
    SmallVector<int64_t> subViewShape;
};

using ExpectedReshapeAndSubViewInputParams = ReshapeAndSubViewInputParams;
using SwapReshapeAndSubViewParams =
        std::tuple<SmallVector<int64_t>, ReshapeAndSubViewInputParams, ExpectedReshapeAndSubViewInputParams>;

class MLIR_ContentSetupTest_SwapReshapeAndSubView :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<SwapReshapeAndSubViewParams> {
protected:
    SmallVector<int64_t> _baseContentShape;
    ReshapeAndSubViewInputParams _inputParams;
    ExpectedReshapeAndSubViewInputParams _expectedParams;

public:
    static std::string getTestCaseName(testing::TestParamInfo<SwapReshapeAndSubViewParams> obj) {
        SmallVector<int64_t> baseContentShape;
        ReshapeAndSubViewInputParams inputParams;
        ExpectedReshapeAndSubViewInputParams expectedParams;
        std::tie(baseContentShape, inputParams, expectedParams) = obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseContentShape={0}_", baseContentShape);
        printTo(result, "inReshape={0}_", inputParams.reshape);
        printTo(result, "inSubView_offset={0}_", inputParams.subViewOffset);
        printTo(result, "inSubView_shape={0}_", inputParams.subViewShape);
        printTo(result, "expectedReshape={0}_", expectedParams.reshape);
        printTo(result, "expectedSubView_offset={0}_", expectedParams.subViewOffset);
        printTo(result, "expectedSubView_shape={0}", expectedParams.subViewShape);

        return result.str();
    }

    void SetUp() override {
        std::tie(_baseContentShape, _inputParams, _expectedParams) = this->GetParam();
    }
};

//
// TEST_SUITE: DoNotSwapReshapeAndSubView
//

using DoNotSwapReshapeAndSubViewParams = std::tuple<SmallVector<int64_t>, ReshapeAndSubViewInputParams>;

class MLIR_ContentSetupTest_DoNotSwapReshapeAndSubView :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<DoNotSwapReshapeAndSubViewParams> {
protected:
    SmallVector<int64_t> _baseContentShape;
    ReshapeAndSubViewInputParams _inputParams;

public:
    static std::string getTestCaseName(testing::TestParamInfo<DoNotSwapReshapeAndSubViewParams> obj) {
        SmallVector<int64_t> baseContentShape;
        ReshapeAndSubViewInputParams inputParams;
        std::tie(baseContentShape, inputParams) = obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseContentShape={0}_", baseContentShape);
        printTo(result, "inReshape={0}_", inputParams.reshape);
        printTo(result, "inSubView_offset={0}_", inputParams.subViewOffset);
        printTo(result, "inSubView_shape={0}", inputParams.subViewShape);

        return result.str();
    }

    void SetUp() override {
        std::tie(_baseContentShape, _inputParams) = this->GetParam();
    }
};

//
// TEST_SUITE: SwapChangeShapeAndElemTypeAndSubView
//

struct QuantCastParams {
    size_t axis;
    size_t numScales;
};

struct ChangeShapeAndElemTypeParams {
    SmallVector<int64_t> shape;
    size_t axis;
    size_t numScales;
};

struct SubViewParams {
    SmallVector<int64_t> offset;
    SmallVector<int64_t> shape;
};

struct QuantCastChangeShapeAndElemTypeAndSubView {
    QuantCastParams quantCast;
    ChangeShapeAndElemTypeParams changeShapeAndElemType;
    SubViewParams subView;
};

struct ExpectedQuantCastParams {
    size_t axis;
    SmallVector<double> scales;
};

struct ExpectedChangeShapeAndElemTypeParams {
    SmallVector<int64_t> shape;
    size_t axis;
    SmallVector<double> scales;
};

struct ExpectedQuantCastChangeShapeAndElemTypeAndSubView {
    ExpectedQuantCastParams quantCast;
    ExpectedChangeShapeAndElemTypeParams changeShapeAndElemType;
    SubViewParams subView;
};

using SwapChangeShapeAndElemTypeAndSubViewParams =
        std::tuple<SmallVector<int64_t>, QuantCastChangeShapeAndElemTypeAndSubView,
                   ExpectedQuantCastChangeShapeAndElemTypeAndSubView>;

class MLIR_ContentSetupTest_SwapChangeShapeAndElemTypeAndSubView :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<SwapChangeShapeAndElemTypeAndSubViewParams> {
protected:
    SmallVector<int64_t> _baseContentShape;
    mlir::quant::QuantizedType _inQuantCastType;
    SmallVector<int64_t> _inChangeShapeAndElemTypeShape;
    mlir::quant::QuantizedType _inChangeShapeAndElemTypeQType;
    SubViewParams _inSubView;

    mlir::quant::QuantizedType _expectedQuantCastType;
    SmallVector<int64_t> _expectedChangeShapeAndElemTypeShape;
    mlir::quant::QuantizedType _expectedChangeShapeAndElemTypeQType;
    SubViewParams _expectedSubView;

public:
    static std::string getTestCaseName(testing::TestParamInfo<SwapChangeShapeAndElemTypeAndSubViewParams> obj) {
        SmallVector<int64_t> baseShape;
        QuantCastChangeShapeAndElemTypeAndSubView inputParams;
        ExpectedQuantCastChangeShapeAndElemTypeAndSubView expectedParams;
        std::tie(baseShape, inputParams, expectedParams) = obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseShape={0}_", baseShape);
        printTo(result, "inQuantCastAxis={0}_", inputParams.quantCast.axis);
        printTo(result, "inQuantCastNScales={0}_", inputParams.quantCast.numScales);
        printTo(result, "inСhangeShapeAndElemTypeNScales={0}_", inputParams.changeShapeAndElemType.numScales);
        printTo(result, "inСhangeShapeAndElemTypeAxis={0}_", inputParams.changeShapeAndElemType.axis);
        printTo(result, "inСhangeShapeAndElemTypeShape={0}_", inputParams.changeShapeAndElemType.shape);
        printTo(result, "inSubViewShape={0}_", inputParams.subView.offset);
        printTo(result, "inSubViewOffset={0}_", inputParams.subView.shape);
        printTo(result, "expectedQuantCastAxis={0}_", expectedParams.quantCast.axis);
        printTo(result, "expectedQuantCastScales={0}_", expectedParams.quantCast.scales);
        printTo(result, "expectedСhangeShapeAndElemTypeScales={0}_", expectedParams.changeShapeAndElemType.scales);
        printTo(result, "expectedСhangeShapeAndElemTypeAxis={0}_", expectedParams.changeShapeAndElemType.axis);
        printTo(result, "expectedСhangeShapeAndElemTypeShape={0}_", expectedParams.changeShapeAndElemType.shape);
        printTo(result, "expectedSubViewShape={0}_", expectedParams.subView.offset);
        printTo(result, "expectedSubViewOffset={0}_", expectedParams.subView.shape);
        return result.str();
    }

    void SetUp() override {
        QuantCastChangeShapeAndElemTypeAndSubView inputParams;
        ExpectedQuantCastChangeShapeAndElemTypeAndSubView expectedParams;
        std::tie(_baseContentShape, inputParams, expectedParams) = this->GetParam();

        ctx.loadDialect<mlir::quant::QuantizationDialect>();
        auto getZeroPoints = [](size_t N) {
            return SmallVector<int64_t>(N, 128);
        };

        auto getScales = [](size_t N) {
            SmallVector<double> scales;
            for (size_t i = 0; i < N; i++) {
                scales.push_back(i + 1);
            }
            return scales;
        };

        _inQuantCastType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), getScales(inputParams.quantCast.numScales),
                getZeroPoints(inputParams.quantCast.numScales), inputParams.quantCast.axis, 0, 255);
        _inChangeShapeAndElemTypeShape = inputParams.changeShapeAndElemType.shape;
        _inChangeShapeAndElemTypeQType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                getScales(inputParams.changeShapeAndElemType.numScales),
                getZeroPoints(inputParams.changeShapeAndElemType.numScales), inputParams.changeShapeAndElemType.axis, 0,
                255);
        _inSubView = inputParams.subView;

        _expectedQuantCastType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), ArrayRef(expectedParams.quantCast.scales),
                getZeroPoints(expectedParams.quantCast.scales.size()), expectedParams.quantCast.axis, 0, 255);
        _expectedChangeShapeAndElemTypeShape = expectedParams.changeShapeAndElemType.shape;
        _expectedChangeShapeAndElemTypeQType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                ArrayRef(expectedParams.changeShapeAndElemType.scales),
                getZeroPoints(expectedParams.changeShapeAndElemType.scales.size()),
                expectedParams.changeShapeAndElemType.axis, 0, 255);
        _expectedSubView = expectedParams.subView;
    }
};

//
// TEST_SUITE: DoNotSwapChangeShapeAndElemTypeAndSubView
//

using DoNotSwapChangeShapeAndElemTypeAndSubViewParams =
        std::tuple<SmallVector<int64_t>, QuantCastChangeShapeAndElemTypeAndSubView>;

class MLIR_ContentSetupTest_DoNotSwapChangeShapeAndElemTypeAndSubView :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<DoNotSwapChangeShapeAndElemTypeAndSubViewParams> {
protected:
    SmallVector<int64_t> _baseContentShape;
    mlir::quant::QuantizedType _inQuantCastType;
    SmallVector<int64_t> _inChangeShapeAndElemTypeShape;
    mlir::quant::QuantizedType _inChangeShapeAndElemTypeQType;
    SubViewParams _inSubView;

public:
    static std::string getTestCaseName(testing::TestParamInfo<DoNotSwapChangeShapeAndElemTypeAndSubViewParams> obj) {
        SmallVector<int64_t> baseShape;
        QuantCastChangeShapeAndElemTypeAndSubView inputParams;
        std::tie(baseShape, inputParams) = obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseShape={0}_", baseShape);
        printTo(result, "inQuantCastAxis={0}_", inputParams.quantCast.axis);
        printTo(result, "inQuantCastNScales={0}_", inputParams.quantCast.numScales);
        printTo(result, "inСhangeShapeAndElemTypeNScales={0}_", inputParams.changeShapeAndElemType.numScales);
        printTo(result, "inСhangeShapeAndElemTypeAxis={0}_", inputParams.changeShapeAndElemType.axis);
        printTo(result, "inСhangeShapeAndElemTypeShape={0}_", inputParams.changeShapeAndElemType.shape);
        printTo(result, "inSubViewShape={0}_", inputParams.subView.offset);
        printTo(result, "inSubViewOffset={0}_", inputParams.subView.shape);
        return result.str();
    }

    void SetUp() override {
        QuantCastChangeShapeAndElemTypeAndSubView inputParams;
        std::tie(_baseContentShape, inputParams) = this->GetParam();

        ctx.loadDialect<mlir::quant::QuantizationDialect>();
        auto getZeroPoints = [](size_t N) {
            return SmallVector<int64_t>(N, 128);
        };

        auto getScales = [](size_t N) {
            SmallVector<double> scales;
            for (size_t i = 0; i < N; i++) {
                scales.push_back(i + 1);
            }
            return scales;
        };

        _inQuantCastType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), getScales(inputParams.quantCast.numScales),
                getZeroPoints(inputParams.quantCast.numScales), inputParams.quantCast.axis, 0, 255);
        _inChangeShapeAndElemTypeShape = inputParams.changeShapeAndElemType.shape;
        _inChangeShapeAndElemTypeQType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                getScales(inputParams.changeShapeAndElemType.numScales),
                getZeroPoints(inputParams.changeShapeAndElemType.numScales), inputParams.changeShapeAndElemType.axis, 0,
                255);
        _inSubView = inputParams.subView;
    }
};

//
// TEST_SUITE: SwapQuantizeTransformationsAndReshape
//

struct QuantizeCastAndChangeTypeParams {
    QuantCastParams inQuantCast;
    QuantCastParams expectedQuantCast;
    QuantCastParams expectedChangeElemType;
};

using OriginShapeType = SmallVector<int64_t>;
using ReshapeType = SmallVector<int64_t>;

using SwapQuantizeTransformationsAndReshapeParams =
        std::tuple<OriginShapeType, ReshapeType, QuantizeCastAndChangeTypeParams>;

class MLIR_ContentSetupTest_SwapQuantizeTransformationsAndReshape :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<SwapQuantizeTransformationsAndReshapeParams> {
protected:
    OriginShapeType _baseContentShape;
    ReshapeType _reshapeShape;
    mlir::quant::QuantizedType _inQuantCastType;
    mlir::quant::QuantizedType _expectedQuantCastType;
    mlir::quant::QuantizedType _expectedChangeElemType;

public:
    static std::string getTestCaseName(testing::TestParamInfo<SwapQuantizeTransformationsAndReshapeParams> obj) {
        OriginShapeType baseContentShape;
        ReshapeType reshapeShape;
        QuantizeCastAndChangeTypeParams quantizeCastAndChangeTypeParams;
        std::tie(baseContentShape, reshapeShape, quantizeCastAndChangeTypeParams) = obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseShape={0}_", baseContentShape);
        printTo(result, "reshapeShape={0}_", reshapeShape);
        printTo(result, "inQuantCastNScales={0}_", quantizeCastAndChangeTypeParams.inQuantCast.numScales);
        printTo(result, "inQuantCastAxis={0}_", quantizeCastAndChangeTypeParams.inQuantCast.axis);
        printTo(result, "expectedQuantCastNScales={0}_", quantizeCastAndChangeTypeParams.expectedQuantCast.numScales);
        printTo(result, "expectedQuantCastAxis={0}_", quantizeCastAndChangeTypeParams.expectedQuantCast.axis);
        printTo(result, "expectedChangeElemTypeNScales={0}_",
                quantizeCastAndChangeTypeParams.expectedChangeElemType.numScales);
        printTo(result, "expectedChangeElemTypeAxis={0}_", quantizeCastAndChangeTypeParams.expectedChangeElemType.axis);
        return result.str();
    }

    void SetUp() override {
        QuantizeCastAndChangeTypeParams quantizeCastAndChangeTypeParams;
        std::tie(_baseContentShape, _reshapeShape, quantizeCastAndChangeTypeParams) = this->GetParam();

        ctx.loadDialect<mlir::quant::QuantizationDialect>();
        auto getZeroPoints = [](size_t N) {
            return SmallVector<int64_t>(N, 128);
        };

        auto getScales = [](size_t N) {
            SmallVector<double> scales;
            for (size_t i = 0; i < N; i++) {
                scales.push_back(i + 1);
            }
            return scales;
        };

        _inQuantCastType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                getScales(quantizeCastAndChangeTypeParams.inQuantCast.numScales),
                getZeroPoints(quantizeCastAndChangeTypeParams.inQuantCast.numScales),
                quantizeCastAndChangeTypeParams.inQuantCast.axis, 0, 255);
        _expectedQuantCastType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                getScales(quantizeCastAndChangeTypeParams.expectedQuantCast.numScales),
                getZeroPoints(quantizeCastAndChangeTypeParams.expectedQuantCast.numScales),
                quantizeCastAndChangeTypeParams.expectedQuantCast.axis, 0, 255);
        _expectedChangeElemType = mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                getScales(quantizeCastAndChangeTypeParams.expectedChangeElemType.numScales),
                getZeroPoints(quantizeCastAndChangeTypeParams.expectedChangeElemType.numScales),
                quantizeCastAndChangeTypeParams.expectedChangeElemType.axis, 0, 255);
    }
};

//
// TEST_SUITE: SwapRelocateWeightsTableAndSubView
//

struct WeightsCompressionParams {
    SmallVector<int64_t> numElems;
};

struct RelocateWeightsTableParams {
    SmallVector<uint32_t> weightsPtr;
    uint64_t sparsityPtr;
    SmallVector<int64_t> offset;
    uint64_t weightsTableSize;
    uint64_t channelOffset;
    std::optional<WeightsCompressionParams> weightsCompression;
};

using ExpectedRelocateWeightsTableParams = RelocateWeightsTableParams;

using SwapRelocateWeightsTableAndSubViewParams =
        std::tuple<SmallVector<int64_t>, SubViewParams, RelocateWeightsTableParams, ExpectedRelocateWeightsTableParams>;

class MLIR_ContentSetupTest_SwapRelocateWeightsTableAndSubView :
        public MLIR_ContentSetupTest,
        public testing::WithParamInterface<SwapRelocateWeightsTableAndSubViewParams> {
protected:
    SmallVector<int64_t> _baseContentShape;
    SubViewParams _subViewParams;
    RelocateWeightsTableParams _relocateWeightsTableParams;

    VPUIP::SparsityCompressionAttr _inSparsityCompression = nullptr;
    Const::RelocateWeightsTableAttr _expectedRelocateWeightsTableAttr = nullptr;

public:
    static std::string getTestCaseName(testing::TestParamInfo<SwapRelocateWeightsTableAndSubViewParams> obj) {
        SmallVector<int64_t> baseContentShape;
        SubViewParams subViewParams;
        RelocateWeightsTableParams relocateWeightsTableParams;
        ExpectedRelocateWeightsTableParams expectedRelocateWeightsTableParams;
        std::tie(baseContentShape, subViewParams, relocateWeightsTableParams, expectedRelocateWeightsTableParams) =
                obj.param;

        std::string str;
        llvm::raw_string_ostream result(str);
        printTo(result, "baseShape={0}_", baseContentShape);
        printTo(result, "subViewOffset={0}_", subViewParams.offset);
        printTo(result, "subViewShape={0}_", subViewParams.shape);
        printTo(result, "inWeightsPtr={0}_", relocateWeightsTableParams.weightsPtr);
        printTo(result, "inSparsityPtr={0}_", relocateWeightsTableParams.sparsityPtr);
        printTo(result, "inOffset={0}_", relocateWeightsTableParams.offset);
        printTo(result, "inWeightsTableSize={0}_", relocateWeightsTableParams.weightsTableSize);
        printTo(result, "inChannelOffset={0}_", relocateWeightsTableParams.channelOffset);
        if (relocateWeightsTableParams.weightsCompression.has_value()) {
            printTo(result, "inWeightsCompressionNumElems={0}_",
                    relocateWeightsTableParams.weightsCompression->numElems);
        }

        printTo(result, "expectedWeightsPtr={0}_", expectedRelocateWeightsTableParams.weightsPtr);
        printTo(result, "expectedSparsityPtr={0}_", expectedRelocateWeightsTableParams.sparsityPtr);
        printTo(result, "expectedOffset={0}_", expectedRelocateWeightsTableParams.offset);
        printTo(result, "expectedWeightsTableSize={0}_", expectedRelocateWeightsTableParams.weightsTableSize);
        printTo(result, "expectedChannelOffset={0}_", expectedRelocateWeightsTableParams.channelOffset);
        if (expectedRelocateWeightsTableParams.weightsCompression.has_value()) {
            printTo(result, "expectedWeightsCompressionNumElems={0}_",
                    expectedRelocateWeightsTableParams.weightsCompression->numElems);
        }
        return result.str();
    }

    void SetUp() override {
        ctx.loadDialect<VPUIP::VPUIPDialect>();

        ExpectedRelocateWeightsTableParams expectedRelocateWeightsTableParams;
        std::tie(_baseContentShape, _subViewParams, _relocateWeightsTableParams, expectedRelocateWeightsTableParams) =
                this->GetParam();

        const auto getSparsityCompression = [&](SmallVector<int64_t> numElems) {
            const int64_t compressionAxis = 0;
            const int64_t alignment = 16;
            const auto numElemsType =
                    mlir::RankedTensorType::get({static_cast<int64_t>(numElems.size())}, getInt64Type(&ctx));
            const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
            return VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis), numElemsAttr,
                                                       getIntAttr(&ctx, alignment));
        };

        if (_relocateWeightsTableParams.weightsCompression.has_value()) {
            _inSparsityCompression =
                    getSparsityCompression(_relocateWeightsTableParams.weightsCompression.value().numElems);
        }

        VPUIP::SparsityCompressionAttr expectedSparsityCompression = nullptr;
        if (expectedRelocateWeightsTableParams.weightsCompression.has_value()) {
            expectedSparsityCompression =
                    getSparsityCompression(expectedRelocateWeightsTableParams.weightsCompression.value().numElems);
        }

        _expectedRelocateWeightsTableAttr = Const::RelocateWeightsTableAttr::get(
                getIntArrayAttr(&ctx, expectedRelocateWeightsTableParams.weightsPtr),
                getIntAttr(&ctx, expectedRelocateWeightsTableParams.sparsityPtr),
                getIntArrayAttr(&ctx, expectedRelocateWeightsTableParams.offset),
                getIntAttr(&ctx, expectedRelocateWeightsTableParams.weightsTableSize), getIntAttr(&ctx, 16),
                expectedSparsityCompression, getIntAttr(&ctx, expectedRelocateWeightsTableParams.channelOffset),
                getIntAttr(&ctx, 0));
    }
};

}  // namespace

TEST_F(MLIR_ContentSetupTest, Invalidated) {
    const auto baseType = mlir::RankedTensorType::get(ArrayRef<int64_t>{4}, getInt8Type(&ctx));
    SmallVector<int8_t> data{1, 7, 10, 15};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(data));
    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    auto setup = contentAttr.transform();

    std::ignore = setup.getContext();
    // move and reassign - setup is valid
    setup = setup.add(1.0);
    // getTransformations() does not invalidate setup
    std::ignore = setup.getTransformations();
    // get() does not invalidate setup
    std::ignore = setup.get();
    // a transformation does invalidate setup
    std::ignore = setup.add(2.0);

    EXPECT_THROW(std::ignore = setup.getContext(), std::exception);
    EXPECT_THROW(std::ignore = setup.addTransformation(nullptr), std::exception);
    EXPECT_THROW(std::ignore = setup.getTransformations(), std::exception);
    EXPECT_THROW(std::ignore = setup.get(), std::exception);
}

TEST_F(MLIR_ContentSetupTest, InvalidBaseContent) {
    EXPECT_THROW(Const::ContentSetup(nullptr), std::exception);
}

//
// FuseConsecutiveTransformations
//

TEST_F(MLIR_ContentSetupTest, FuseSubView) {
    const int64_t IC = 1;
    const int64_t IH = 7;
    const int64_t IW = 5;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first SubView
    SmallVector<int64_t> offset1 = {0, 1, 2};
    SmallVector<int64_t> shape1 = {1, 6, 3};

    auto contentAttrSetup = baseContentAttrSetup.subview(ShapeRef(offset1), ShapeRef(shape1));

    // second SubView
    SmallVector<int64_t> offset2 = {0, 1, 1};
    SmallVector<int64_t> shape2 = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(offset2), ShapeRef(shape2));

    // final SubView params
    SmallVector<int64_t> offsetFinal = {0, 2, 3};
    SmallVector<int64_t> shapeFinal = {1, 2, 1};

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkSubViewAttr(actualTransformations[0], offsetFinal, shapeFinal);
}

TEST_F(MLIR_ContentSetupTest, FuseAdd) {
    const int64_t IC = 1;
    const int64_t IH = 7;
    const int64_t IW = 5;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first add
    auto contentAttrSetup = baseContentAttrSetup.add(1);

    // second add
    contentAttrSetup = contentAttrSetup.add(2);

    // check final add
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkAddAttr(actualTransformations[0], 3);
}

TEST_F(MLIR_ContentSetupTest, FuseRescale) {
    const int64_t IC = 1;
    const int64_t IH = 7;
    const int64_t IW = 5;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first rescale
    auto contentAttrSetup = baseContentAttrSetup.rescale(2);

    // second rescale
    contentAttrSetup = contentAttrSetup.rescale(3);

    // check final rescale
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkRescaleAttr(actualTransformations[0], 6);
}

TEST_F(MLIR_ContentSetupTest, FuseReshape) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first reshape
    auto contentAttrSetup = baseContentAttrSetup.reshape({IC, IH * IW});

    // second reshape
    SmallVector<int64_t> expectedShape = {2, 4, 12};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check final reshape
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkReshapeAttr(actualTransformations[0], ShapeRef(expectedShape));
}

TEST_F(MLIR_ContentSetupTest, FuseReorder) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first reorder
    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::WHC);

    // second reorder
    contentAttrSetup = contentAttrSetup.reorder(DimsOrder::HCW);

    // check final reorder
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkReorderAttr(actualTransformations[0], DimsOrder::HCW);
}

TEST_F(MLIR_ContentSetupTest, FuseCastElemType) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first cast
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::Float16Type::get(&ctx));

    // second cast
    const auto expectedType = getUInt8Type(&ctx);
    contentAttrSetup = contentAttrSetup.castElemType(expectedType);

    // check final cast
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkCastElemTypeAttr(actualTransformations[0], expectedType);
}

TEST_F(MLIR_ContentSetupTest, FuseCastElemType_Quantized) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, mlir::Float64Type::get(&ctx));

    // hide the storage type
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::Float32Type::get(&ctx));

    // cast to quantized type
    const auto quantizedType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx),
                                                                      mlir::Float32Type::get(&ctx), 0.078, 128, 0, 255);
    contentAttrSetup = contentAttrSetup.castElemType(quantizedType);

    // check casts
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    // Note: no fusing in this case
    checkCastElemTypeAttr(actualTransformations[0], mlir::Float32Type::get(&ctx));
    checkCastElemTypeAttr(actualTransformations[1], quantizedType);
}

//
// MoveSubViewBefore
//

TEST_F(MLIR_ContentSetupTest, SwapAddAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Add
    auto contentAttrSetup = baseContentAttrSetup.add(1);

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkAddAttr(actualTransformations[1], 1);
}

TEST_F(MLIR_ContentSetupTest, SwapReorderAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Reorder
    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::WHC);

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkReorderAttr(actualTransformations[1], DimsOrder::WHC);
}

TEST_F(MLIR_ContentSetupTest, SwapMemPermuteAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first MemPermute
    auto contentAttrSetup = baseContentAttrSetup.memPermute(DimsOrder::HWC, DimsOrder::CHW);

    // second SubView
    SmallVector<int64_t> initialOffset = {0, 0, 6};
    SmallVector<int64_t> initialShape = {3, 4, 2};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(initialOffset), ShapeRef(initialShape));

    SmallVector<int64_t> expectedOffset = {0, 6, 0};
    SmallVector<int64_t> expectedShape = {4, 2, 3};

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkMemPermuteAttr(actualTransformations[1], DimsOrder::HWC, DimsOrder::CHW);
}

TEST_F(MLIR_ContentSetupTest, SwapTransposeAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Transpose
    auto contentAttrSetup = baseContentAttrSetup.transpose(DimsOrder::WHC);

    // second SubView
    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 2;

    const int64_t OC = 3;
    const int64_t OH = 2;
    const int64_t OW = 1;

    contentAttrSetup = contentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], {OFF_W, OFF_H, OFF_C}, {OW, OH, OC});
    checkTransposeAttr(actualTransformations[1], DimsOrder::WHC);
}

TEST_F(MLIR_ContentSetupTest, SwapRescaleAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Rescale
    auto contentAttrSetup = baseContentAttrSetup.rescale(3);

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkRescaleAttr(actualTransformations[1], 3);
}

TEST_F(MLIR_ContentSetupTest, DontSwapSubviewWithRangeIntoPadAfterRegion) {
    const int64_t IC = 1;
    const int64_t IH = 1;
    const int64_t IW = 256;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first PadWithZero
    SmallVector<int64_t> expectedPadBegin = {0, 0, 0};
    SmallVector<int64_t> expectedPadEnd = {0, 0, 256};

    auto contentAttrSetup = baseContentAttrSetup.padWithZero(ShapeRef(expectedPadBegin), ShapeRef(expectedPadEnd));

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 0, 256};
    SmallVector<int64_t> expectedShape = {1, 1, 128};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkPadWithZeroAttr(actualTransformations[0], expectedPadBegin, expectedPadEnd);
    checkSubViewAttr(actualTransformations[1], expectedOffset, expectedShape);
}

TEST_F(MLIR_ContentSetupTest, DontSwapSubviewWithRangeIntoPadBeforeRegion) {
    const int64_t IC = 1;
    const int64_t IH = 1;
    const int64_t IW = 256;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first PadWithZero
    SmallVector<int64_t> expectedPadBegin = {0, 0, 256};
    SmallVector<int64_t> expectedPadEnd = {0, 0, 0};

    auto contentAttrSetup = baseContentAttrSetup.padWithZero(ShapeRef(expectedPadBegin), ShapeRef(expectedPadEnd));

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 0, 0};
    SmallVector<int64_t> expectedShape = {1, 1, 256};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkPadWithZeroAttr(actualTransformations[0], expectedPadBegin, expectedPadEnd);
    checkSubViewAttr(actualTransformations[1], expectedOffset, expectedShape);
}

TEST_F(MLIR_ContentSetupTest, SwapSubviewCoveringOriginalConstantAndPaddedRegionsWithPadWithZero) {
    const int64_t IC = 1;
    const int64_t IH = 8;
    const int64_t IW = 256;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first PadWithZero
    SmallVector<int64_t> padBegin = {0, 0, 128};
    SmallVector<int64_t> padEnd = {0, 0, 128};

    auto contentAttrSetup = baseContentAttrSetup.padWithZero(ShapeRef(padBegin), ShapeRef(padEnd));

    // second SubView
    SmallVector<int64_t> offset = {0, 2, 64};
    SmallVector<int64_t> shape = {1, 4, 384};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(offset), ShapeRef(shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    SmallVector<int64_t> expectedSubViewOffset = {0, 2, 0};
    SmallVector<int64_t> expectedSubViewShape = {1, 4, 256};
    checkSubViewAttr(actualTransformations[0], expectedSubViewOffset, expectedSubViewShape);

    SmallVector<int64_t> expectedPadBefore = {0, 0, 64};
    SmallVector<int64_t> expectedPadAfter = {0, 0, 64};
    checkPadWithZeroAttr(actualTransformations[1], expectedPadBefore, expectedPadAfter);
}

TEST_F(MLIR_ContentSetupTest, SwapSubviewCoveringOnlyPartOfOriginalConstantWithPadWithZero) {
    const int64_t IC = 1;
    const int64_t IH = 8;
    const int64_t IW = 256;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first PadWithZero
    SmallVector<int64_t> padBegin = {0, 0, 128};
    SmallVector<int64_t> padEnd = {0, 0, 128};

    auto contentAttrSetup = baseContentAttrSetup.padWithZero(ShapeRef(padBegin), ShapeRef(padEnd));

    // second SubView
    SmallVector<int64_t> offset = {0, 2, 192};
    SmallVector<int64_t> shape = {1, 4, 64};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(offset), ShapeRef(shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    SmallVector<int64_t> expectedSubViewOffset = {0, 2, 64};
    SmallVector<int64_t> expectedSubViewShape = {1, 4, 64};
    checkSubViewAttr(actualTransformations[0], expectedSubViewOffset, expectedSubViewShape);
}

TEST_F(MLIR_ContentSetupTest, SwapAndFoldSubViewWithPadWithZero) {
    const int64_t IC = 1;
    const int64_t IH = 8;
    const int64_t IW = 256;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first PadWithZero
    SmallVector<int64_t> padBegin = {0, 0, 0};
    SmallVector<int64_t> padEnd = {0, 8, 0};

    auto contentAttrSetup = baseContentAttrSetup.padWithZero(ShapeRef(padBegin), ShapeRef(padEnd));

    // second SubView
    SmallVector<int64_t> offset = {0, 0, 0};
    SmallVector<int64_t> shape = {1, 8, 256};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(offset), ShapeRef(shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 0);
}

TEST_F(MLIR_ContentSetupTest, SwapMemPermAndSubViewMultipleInstances) {
    const int64_t IN = 4096;
    const int64_t IC = 11008;
    const int64_t IH = 1;
    const int64_t IW = 1;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IN, IC, IH, IW}, getInt8Type(&ctx));

    // 1st Transpose
    auto contentAttrSetup = baseContentAttrSetup.transpose(DimsOrder::HCNW);

    // +2 SubViews
    contentAttrSetup = contentAttrSetup.subview({0, 8192, 0, 0}, {1, 2816, 4096, 1});
    contentAttrSetup = contentAttrSetup.subview({0, 0, 0, 0}, {1, 2816, 320, 1});

    // + Reorder
    contentAttrSetup = contentAttrSetup.reorder(DimsOrder::NHWC);

    // +2 SubViews
    contentAttrSetup = contentAttrSetup.subview({0, 1000, 0, 0}, {1, 1816, 320, 1});
    contentAttrSetup = contentAttrSetup.subview({0, 0, 0, 0}, {1, 1000, 320, 1});

    // + Reshape
    SmallVector<int64_t> expectedShape = {10, 100, 32, 10};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 4);

    SmallVector<int64_t> expectedSubViewOffset = {0, 9192, 0, 0};
    SmallVector<int64_t> expectedSubViewShape = {320, 1000, 1, 1};
    checkSubViewAttr(actualTransformations[0], expectedSubViewOffset, expectedSubViewShape);
    checkTransposeAttr(actualTransformations[1], DimsOrder::HCNW);
    checkReorderAttr(actualTransformations[2], DimsOrder::NHWC);
    checkReshapeAttr(actualTransformations[3], expectedShape);
}

TEST_F(MLIR_ContentSetupTest, SwapCastElemTypeAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, mlir::Float32Type::get(&ctx));

    // first CastElemType
    auto expectedType = mlir::Float16Type::get(&ctx);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::Float16Type::get(&ctx));

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkCastElemTypeAttr(actualTransformations[1], expectedType);
}

TEST_F(MLIR_ContentSetupTest, SwapConvertElemTypeAndSubView) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, mlir::Float32Type::get(&ctx));

    // first ConvertElemType
    auto expectedType = mlir::Float16Type::get(&ctx);
    auto contentAttrSetup = baseContentAttrSetup.convertElemType(mlir::Float16Type::get(&ctx));

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkConvertElemTypeAttr(actualTransformations[1], expectedType);
}

TEST_F(MLIR_ContentSetupTest, SwapConvertElemTypeAndSubView_Subbyte) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getUInt4Type(&ctx));

    // first ConvertElemType
    auto expectedType = getUInt8Type(&ctx);
    auto contentAttrSetup = baseContentAttrSetup.convertElemType(expectedType);

    // second SubView
    SmallVector<int64_t> expectedOffset = {0, 1, 1};
    SmallVector<int64_t> expectedShape = {1, 2, 1};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    // Note: subview supports sub-byte types, so swapping is legal - see
    // TEST_F(MLIR_SubByteTest, SubViewI4_General)
    checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
    checkConvertElemTypeAttr(actualTransformations[1], expectedType);
}

TEST_F(MLIR_ContentSetupTest, SwapConvertElemTypeAndSubView_Quantized) {
    const int64_t IC = 2;
    const int64_t IH = 8;
    const int64_t IW = 3;

    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    auto perTensorQType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                 0.078, 128, 0, 255);
    auto perTensorQType2 =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), 0.078, 0, -128, 127);

    auto perAxisQType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), {2, 0.5}, {127, 127}, 0, 0, 255);
    auto perAxisQType2 =
            mlir::quant::UniformQuantizedPerAxisType::get(mlir::quant::QuantizationFlags::Signed, getInt8Type(&ctx),
                                                          mlir::Float32Type::get(&ctx), {2, 0.5}, {0, 0}, 0, -128, 127);
    auto expectedPerAxisQType =
            mlir::quant::UniformQuantizedPerAxisType::get(mlir::quant::QuantizationFlags::Signed, getInt8Type(&ctx),
                                                          mlir::Float32Type::get(&ctx), {2}, {0}, 0, -128, 127);

    SmallVector<std::tuple<mlir::quant::QuantizedType, mlir::quant::QuantizedType, mlir::quant::QuantizedType>> qTypes{
            {perTensorQType, perTensorQType2, perTensorQType2},
            {perAxisQType, perAxisQType2, expectedPerAxisQType}};

    for (auto [qTypeIn, qTypeOut, qTypeExpected] : qTypes) {
        auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getUInt8Type(&ctx));

        // first cast to quantized (required by convert to succeed)
        auto contentAttrSetup = baseContentAttrSetup.castElemType(qTypeIn);

        // convert from one quantized to another
        contentAttrSetup = contentAttrSetup.convertElemType(qTypeOut);

        // then add subview
        SmallVector<int64_t> expectedOffset = {0, 1, 1};
        SmallVector<int64_t> expectedShape = {1, 2, 1};
        contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

        // check
        auto actualTransformations = contentAttrSetup.getTransformations();
        EXPECT_EQ(actualTransformations.size(), 3);

        checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
        checkConvertElemTypeAttr(actualTransformations[2], qTypeExpected);
    }
}

TEST_F(MLIR_ContentSetupTest, SwapCastElemTypeAndSubView_Quantized) {
    const int64_t IC = 2;
    const int64_t IH = 8;
    const int64_t IW = 3;

    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    auto perTensorQType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                 0.078, 128, 0, 255);

    auto perAxisQType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), {2, 0.5}, {127, 127}, 0, 0, 255);
    auto expectedPerAxisQType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), {2}, {127}, 0, 0, 255);

    SmallVector<std::pair<mlir::quant::QuantizedType, mlir::quant::QuantizedType>> qTypes{
            {perTensorQType, perTensorQType},
            {perAxisQType, expectedPerAxisQType}};

    for (auto& qType : qTypes) {
        auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getUInt8Type(&ctx));

        // first QuantCast
        auto contentAttrSetup = baseContentAttrSetup.castElemType(qType.first);

        // second SubView
        SmallVector<int64_t> expectedOffset = {0, 1, 1};
        SmallVector<int64_t> expectedShape = {1, 2, 1};

        contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedOffset), ShapeRef(expectedShape));

        // check
        auto actualTransformations = contentAttrSetup.getTransformations();
        EXPECT_EQ(actualTransformations.size(), 2);

        checkSubViewAttr(actualTransformations[0], expectedOffset, expectedShape);
        checkCastElemTypeAttr(actualTransformations[1], qType.second);
    }
}

TEST_P(MLIR_ContentSetupTest_SwapReshapeAndSubView, SwapReshapeAndSubView) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{_baseContentShape}, getInt8Type(&ctx));

    // first Reshape
    auto contentAttrSetup = baseContentAttrSetup.reshape(ShapeRef(_inputParams.reshape));

    // second SubView
    contentAttrSetup =
            contentAttrSetup.subview(ShapeRef(_inputParams.subViewOffset), ShapeRef(_inputParams.subViewShape));

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], _expectedParams.subViewOffset, _expectedParams.subViewShape);
    checkReshapeAttr(actualTransformations[1], _expectedParams.reshape);
}

SmallVector<SwapReshapeAndSubViewParams, 5> swapReshapeAndSubViewParams = {
        SwapReshapeAndSubViewParams{
                {1, 16, 1, 8}, {{16, 8, 1, 1}, {0, 0, 0, 0}, {8, 8, 1, 1}}, {{8, 8, 1, 1}, {0, 0, 0, 0}, {1, 8, 1, 8}}},
        SwapReshapeAndSubViewParams{
                {1, 16, 1, 8}, {{16, 8, 1, 1}, {8, 0, 0, 0}, {8, 8, 1, 1}}, {{8, 8, 1, 1}, {0, 8, 0, 0}, {1, 8, 1, 8}}},
        SwapReshapeAndSubViewParams{
                {1, 16, 8, 8}, {{1, 16, 64}, {0, 0, 0}, {1, 8, 64}}, {{1, 8, 64}, {0, 0, 0, 0}, {1, 8, 8, 8}}},
        SwapReshapeAndSubViewParams{
                {1, 16, 8}, {{1, 16, 2, 4}, {0, 0, 0, 0}, {1, 8, 2, 4}}, {{1, 8, 2, 4}, {0, 0, 0}, {1, 8, 8}}},
        SwapReshapeAndSubViewParams{{1, 64, 32, 16, 3, 3},
                                    {{64, 32, 16, 9}, {16, 8, 0, 0}, {48, 24, 16, 9}},
                                    {{48, 24, 16, 9}, {0, 16, 8, 0, 0, 0}, {1, 48, 24, 16, 3, 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_SwapReshapeAndSubView, MLIR_ContentSetupTest_SwapReshapeAndSubView,
                         ::testing::ValuesIn(swapReshapeAndSubViewParams),
                         MLIR_ContentSetupTest_SwapReshapeAndSubView::getTestCaseName);

TEST_P(MLIR_ContentSetupTest_DoNotSwapReshapeAndSubView, DoNotSwapReshapeAndSubView) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{_baseContentShape}, getInt8Type(&ctx));

    // first Reshape
    auto contentAttrSetup = baseContentAttrSetup.reshape(ShapeRef(_inputParams.reshape));

    // second SubView
    contentAttrSetup =
            contentAttrSetup.subview(ShapeRef(_inputParams.subViewOffset), ShapeRef(_inputParams.subViewShape));

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkReshapeAttr(actualTransformations[0], _inputParams.reshape);
    checkSubViewAttr(actualTransformations[1], _inputParams.subViewOffset, _inputParams.subViewShape);
}

SmallVector<DoNotSwapReshapeAndSubViewParams> doNotswapReshapeAndSubViewParams = {
        DoNotSwapReshapeAndSubViewParams{{1, 16, 1, 8}, {{128, 1}, {0, 0}, {64, 1}}},
        DoNotSwapReshapeAndSubViewParams{{1, 16, 48, 1}, {{1, 16, 12, 4}, {0, 0, 9, 0}, {1, 16, 3, 4}}}};

INSTANTIATE_TEST_SUITE_P(smoke_DoNotSwapReshapeAndSubView, MLIR_ContentSetupTest_DoNotSwapReshapeAndSubView,
                         ::testing::ValuesIn(doNotswapReshapeAndSubViewParams),
                         MLIR_ContentSetupTest_DoNotSwapReshapeAndSubView::getTestCaseName);

TEST_P(MLIR_ContentSetupTest_SwapChangeShapeAndElemTypeAndSubView, SwapChangeShapeAndElemTypeAndSubView) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef(_baseContentShape), getUInt8Type(&ctx));

    // first QuantCast
    auto contentAttrSetup = baseContentAttrSetup.castElemType(_inQuantCastType);

    // second ChangeShapeAndElemType
    contentAttrSetup = contentAttrSetup.changeShapeAndElemType(ShapeRef(_inChangeShapeAndElemTypeShape),
                                                               _inChangeShapeAndElemTypeQType);

    // third SubView
    contentAttrSetup = contentAttrSetup.subview(ShapeRef(_inSubView.offset), ShapeRef(_inSubView.shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 3);

    checkSubViewAttr(actualTransformations[0], _expectedSubView.offset, _expectedSubView.shape);
    checkCastElemTypeAttr(actualTransformations[1], _expectedQuantCastType);
    checkChangeShapeAndElemTypeAttr(actualTransformations[2], _expectedChangeShapeAndElemTypeShape,
                                    _expectedChangeShapeAndElemTypeQType);
}

SmallVector<SwapChangeShapeAndElemTypeAndSubViewParams, 5> swapChangeShapeAndElemTypeAndSubViewParams = {
        SwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 4, 1, 2},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 4}, {{4, 2, 1, 1}, 0, 4}, {{0, 0, 0, 0}, {3, 2, 1, 1}}},
                ExpectedQuantCastChangeShapeAndElemTypeAndSubView{
                        {1, {1.0, 2.0, 3.0}}, {{3, 2, 1, 1}, 0, {1.0, 2.0, 3.0}}, {{0, 0, 0, 0}, {1, 3, 1, 2}}}},
        SwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 4, 1, 2},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 4}, {{4, 2, 1, 1}, 0, 4}, {{1, 0, 0, 0}, {2, 2, 1, 1}}},
                ExpectedQuantCastChangeShapeAndElemTypeAndSubView{
                        {1, {2.0, 3.0}}, {{2, 2, 1, 1}, 0, {2.0, 3.0}}, {{0, 1, 0, 0}, {1, 2, 1, 2}}}},
        SwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 4, 2, 2},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 4}, {{1, 4, 4}, 1, 4}, {{0, 0, 0}, {1, 2, 4}}},
                ExpectedQuantCastChangeShapeAndElemTypeAndSubView{
                        {1, {1.0, 2.0}}, {{1, 2, 4}, 1, {1.0, 2.0}}, {{0, 0, 0, 0}, {1, 2, 2, 2}}}},
        SwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 8, 4},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 8}, {{1, 8, 2, 2}, 1, 8}, {{0, 1, 0, 0}, {1, 4, 2, 2}}},
                ExpectedQuantCastChangeShapeAndElemTypeAndSubView{
                        {1, {2.0, 3.0, 4.0, 5.0}}, {{1, 4, 2, 2}, 1, {2.0, 3.0, 4.0, 5.0}}, {{0, 1, 0}, {1, 4, 4}}}},
        SwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 8, 6, 4, 3, 3},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 8}, {{8, 6, 4, 9}, 0, 8}, {{2, 1, 0, 0}, {4, 3, 4, 9}}},
                ExpectedQuantCastChangeShapeAndElemTypeAndSubView{{1, {3.0, 4.0, 5.0, 6.0}},
                                                                  {{4, 3, 4, 9}, 0, {3.0, 4.0, 5.0, 6.0}},
                                                                  {{0, 2, 1, 0, 0, 0}, {1, 4, 3, 4, 3, 3}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_SwapChangeShapeAndElemTypeAndSubView,
                         MLIR_ContentSetupTest_SwapChangeShapeAndElemTypeAndSubView,
                         ::testing::ValuesIn(swapChangeShapeAndElemTypeAndSubViewParams),
                         MLIR_ContentSetupTest_SwapChangeShapeAndElemTypeAndSubView::getTestCaseName);

TEST_P(MLIR_ContentSetupTest_DoNotSwapChangeShapeAndElemTypeAndSubView, DoNotSwapChangeShapeAndElemTypeAndSubView) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef(_baseContentShape), getUInt8Type(&ctx));

    // first QuantCast
    auto contentAttrSetup = baseContentAttrSetup.castElemType(_inQuantCastType);

    // second ChangeShapeAndElemType
    contentAttrSetup = contentAttrSetup.changeShapeAndElemType(ShapeRef(_inChangeShapeAndElemTypeShape),
                                                               _inChangeShapeAndElemTypeQType);

    // third SubView
    contentAttrSetup = contentAttrSetup.subview(ShapeRef(_inSubView.offset), ShapeRef(_inSubView.shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 3);

    checkCastElemTypeAttr(actualTransformations[0], _inQuantCastType);
    checkChangeShapeAndElemTypeAttr(actualTransformations[1], _inChangeShapeAndElemTypeShape,
                                    _inChangeShapeAndElemTypeQType);
    checkSubViewAttr(actualTransformations[2], _inSubView.offset, _inSubView.shape);
}

SmallVector<DoNotSwapChangeShapeAndElemTypeAndSubViewParams, 5> doNotSwapChangeShapeAndElemTypeAndSubViewParams = {
        DoNotSwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 4, 1, 2},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 4}, {{8, 1}, 0, 8}, {{0, 0}, {4, 1}}}},
        DoNotSwapChangeShapeAndElemTypeAndSubViewParams{
                SmallVector<int64_t>{1, 8, 8, 2},
                QuantCastChangeShapeAndElemTypeAndSubView{{1, 8}, {{1, 8, 4, 4}, 1, 8}, {{0, 0, 3, 0}, {1, 8, 1, 4}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_DoNotSwapChangeShapeAndElemTypeAndSubView,
                         MLIR_ContentSetupTest_DoNotSwapChangeShapeAndElemTypeAndSubView,
                         ::testing::ValuesIn(doNotSwapChangeShapeAndElemTypeAndSubViewParams),
                         MLIR_ContentSetupTest_DoNotSwapChangeShapeAndElemTypeAndSubView::getTestCaseName);

TEST_F(MLIR_ContentSetupTest, SwapMultipleTransformationsAndSubView) {
    SmallVector<int64_t> baseContentShape = {4, 8, 3};
    auto baseContentAttrSetup = getContentSetup(ArrayRef(baseContentShape), mlir::Float32Type::get(&ctx));

    size_t numOfAddRescalePairs = 10;
    for (size_t i = 0; i < numOfAddRescalePairs; i++) {
        baseContentAttrSetup = baseContentAttrSetup.add(1 + i);
        baseContentAttrSetup = baseContentAttrSetup.rescale(2 + i);
    }

    // SubView
    SmallVector<int64_t> expectedSubViewOffset = {0, 1, 1};
    SmallVector<int64_t> expectedSubViewShape = {1, 4, 2};

    baseContentAttrSetup =
            baseContentAttrSetup.subview(ShapeRef(expectedSubViewOffset), ShapeRef(expectedSubViewShape));

    // check
    auto actualTransformations = baseContentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), numOfAddRescalePairs * 2 + 1);

    checkSubViewAttr(actualTransformations[0], expectedSubViewOffset, expectedSubViewShape);

    for (size_t i = 0; i < numOfAddRescalePairs; i++) {
        checkAddAttr(actualTransformations[i * 2 + 1], 1 + i);
        checkRescaleAttr(actualTransformations[i * 2 + 2], 2 + i);
    }
}

//
// MoveReshapeBefore
//

TEST_F(MLIR_ContentSetupTest, SwapAddAndReshape) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Add
    auto contentAttrSetup = baseContentAttrSetup.add(1);

    // second SubView
    SmallVector<int64_t> expectedShape = {4, 4, 6};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkReshapeAttr(actualTransformations[0], expectedShape);
    checkAddAttr(actualTransformations[1], 1);
}

TEST_F(MLIR_ContentSetupTest, SwapRescaleAndReshape) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    // first Rescale
    auto contentAttrSetup = baseContentAttrSetup.rescale(3);

    // second SubView
    SmallVector<int64_t> expectedShape = {4, 4, 6};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkReshapeAttr(actualTransformations[0], expectedShape);
    checkRescaleAttr(actualTransformations[1], 3);
}

TEST_F(MLIR_ContentSetupTest, SwapCastElemTypeAndReshape) {
    const int64_t IC = 4;
    const int64_t IH = 8;
    const int64_t IW = 3;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, mlir::Float32Type::get(&ctx));

    // first CastElemType
    auto expectedType = mlir::Float16Type::get(&ctx);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::Float16Type::get(&ctx));

    // second SubView
    SmallVector<int64_t> expectedShape = {1, 2, 1};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkReshapeAttr(actualTransformations[0], expectedShape);
    checkCastElemTypeAttr(actualTransformations[1], expectedType);
}

TEST_P(MLIR_ContentSetupTest_SwapQuantizeTransformationsAndReshape, SwapQuantizeTransformationsAndReshape) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef(_baseContentShape), getUInt8Type(&ctx));

    // first QuantCast
    auto contentAttrSetup = baseContentAttrSetup.castElemType(_inQuantCastType);

    // second Dequantize
    contentAttrSetup = contentAttrSetup.dequantize();

    // third SubView
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(_reshapeShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 3);

    checkCastElemTypeAttr(actualTransformations[0], _expectedQuantCastType);
    checkChangeShapeAndElemTypeAttr(actualTransformations[1], _reshapeShape, _expectedChangeElemType);
    checkDequantizeAttr(actualTransformations[2]);
}

SmallVector<SwapQuantizeTransformationsAndReshapeParams, 5> swapQuantizeTransformationsAndReshapeParams = {
        SwapQuantizeTransformationsAndReshapeParams{{2, 3, 1, 1}, {1, 2, 3, 1}, {{1, 3}, {1, 3}, {2, 3}}},
        SwapQuantizeTransformationsAndReshapeParams{{1, 2, 3, 1}, {2, 1, 3, 1}, {{2, 3}, {2, 3}, {2, 3}}},
        SwapQuantizeTransformationsAndReshapeParams{{1, 3, 1, 2}, {3, 2, 1, 1}, {{1, 3}, {1, 3}, {0, 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_SwapQuantizeTransformationsAndReshape,
                         MLIR_ContentSetupTest_SwapQuantizeTransformationsAndReshape,
                         ::testing::ValuesIn(swapQuantizeTransformationsAndReshapeParams),
                         MLIR_ContentSetupTest_SwapQuantizeTransformationsAndReshape::getTestCaseName);

TEST_F(MLIR_ContentSetupTest, SwapPerTensorQuantizeTransformationsAndReshape) {
    SmallVector<int64_t> baseContentShape = {16, 32, 1, 1};
    auto baseContentAttrSetup = getContentSetup(ArrayRef(baseContentShape), getUInt8Type(&ctx));

    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    auto perTensorQType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                 0.078, 128, 0, 255);
    // first QuantCast
    auto contentAttrSetup = baseContentAttrSetup.castElemType(perTensorQType);

    // second Dequantize
    contentAttrSetup = contentAttrSetup.dequantize();

    // third SubView
    SmallVector<int64_t> reshapeShape = {16, 8, 2, 2};
    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(reshapeShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 3);

    checkCastElemTypeAttr(actualTransformations[0], perTensorQType);
    checkReshapeAttr(actualTransformations[1], reshapeShape);
    checkDequantizeAttr(actualTransformations[2]);
}

TEST_P(MLIR_ContentSetupTest_SwapRelocateWeightsTableAndSubView, SwapRelocateWeightsTableAndSubView) {
    auto baseContentAttrSetup = getContentSetup(ArrayRef(_baseContentShape), getSInt32Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.relocateWeightsTablePointers(
            _relocateWeightsTableParams.weightsPtr, _relocateWeightsTableParams.sparsityPtr,
            ShapeRef(_relocateWeightsTableParams.offset), _relocateWeightsTableParams.weightsTableSize,
            /* weightsElemBitSize = */ 16, _inSparsityCompression, _relocateWeightsTableParams.channelOffset, 0);

    // second SubView
    contentAttrSetup = contentAttrSetup.subview(ShapeRef(_subViewParams.offset), ShapeRef(_subViewParams.shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    checkSubViewAttr(actualTransformations[0], _subViewParams.offset, _subViewParams.shape);
    checkRelocateWeightsTableAttr(actualTransformations[1], _expectedRelocateWeightsTableAttr);
}

SmallVector<SwapRelocateWeightsTableAndSubViewParams, 5> swapRelocateWeightsTableAndSubView = {
        SwapRelocateWeightsTableAndSubViewParams{{4, 1, 1, 4},
                                                 {{0, 0, 0, 0}, {2, 1, 1, 4}},
                                                 {{100}, 16777215, {0}, 64, 0, std::nullopt},
                                                 {{100}, 16777215, {0}, 32, 0, std::nullopt}},
        SwapRelocateWeightsTableAndSubViewParams{{4, 1, 1, 4},
                                                 {{1, 0, 0, 0}, {2, 1, 1, 4}},
                                                 {{100}, 16777215, {0}, 64, 0, std::nullopt},
                                                 {{100}, 16777215, {0}, 32, 1, std::nullopt}},
        SwapRelocateWeightsTableAndSubViewParams{{4, 1, 1, 4},
                                                 {{2, 0, 0, 0}, {2, 1, 1, 4}},
                                                 {{100}, 200, {0}, 64, 0, WeightsCompressionParams{{1, 2, 3, 4}}},
                                                 {{100}, 200, {0}, 32, 2, WeightsCompressionParams{{1, 2, 3, 4}}}},
        SwapRelocateWeightsTableAndSubViewParams{{4, 1, 1, 4},
                                                 {{2, 0, 0, 0}, {2, 1, 1, 4}},
                                                 {{100}, 200, {0}, 64, 2, WeightsCompressionParams{{1, 2, 3, 4}}},
                                                 {{100}, 200, {0}, 32, 4, WeightsCompressionParams{{1, 2, 3, 4}}}},
        SwapRelocateWeightsTableAndSubViewParams{{8, 1, 1, 4},
                                                 {{4, 0, 0, 0}, {2, 1, 1, 4}},
                                                 {{100, 200, 300, 400}, 16777215, {0, 2, 4, 6}, 128, 0, std::nullopt},
                                                 {{300}, 16777215, {0}, 32, 0, std::nullopt}},
        SwapRelocateWeightsTableAndSubViewParams{
                {8, 1, 1, 4},
                {{4, 0, 0, 0}, {2, 1, 1, 4}},
                {{100, 200, 300, 400}, 200, {0, 2, 4, 6}, 128, 0, WeightsCompressionParams{{1, 2, 3, 4, 5, 6, 7, 8}}},
                {{300}, 200, {0}, 32, 0, WeightsCompressionParams{{5, 6}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_SwapRelocateWeightsTableAndSubView,
                         MLIR_ContentSetupTest_SwapRelocateWeightsTableAndSubView,
                         ::testing::ValuesIn(swapRelocateWeightsTableAndSubView),
                         MLIR_ContentSetupTest_SwapRelocateWeightsTableAndSubView::getTestCaseName);

TEST_F(MLIR_ContentSetupTest, DoNotSwapRelocateWeightsTableAndSubView) {
    SmallVector<int64_t> baseContentShape = {8, 1, 1, 4};
    auto baseContentAttrSetup = getContentSetup(ArrayRef(baseContentShape), getSInt32Type(&ctx));

    SmallVector<uint32_t> weightsPtr = {100, 200, 300, 400};
    auto sparsityPtr = 16777215;
    SmallVector<int64_t> offsets = {0, 2, 4, 6};
    auto weightsTableSize = 128;
    auto weightsElemBitSize = 16;
    auto channelOffset = 0;

    auto contentAttrSetup = baseContentAttrSetup.relocateWeightsTablePointers(
            weightsPtr, sparsityPtr, ShapeRef(offsets), weightsTableSize, weightsElemBitSize, nullptr, channelOffset,
            0);

    // second SubView
    SmallVector<int64_t> offset = {3, 0, 0, 0};
    SmallVector<int64_t> shape = {2, 1, 1, 4};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(offset), ShapeRef(shape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);

    auto expectedRelocateWeightsTableAttr = Const::RelocateWeightsTableAttr::get(
            getIntArrayAttr(&ctx, ArrayRef(weightsPtr)), getIntAttr(&ctx, sparsityPtr),
            getIntArrayAttr(&ctx, ArrayRef(offsets)), getIntAttr(&ctx, weightsTableSize),
            getIntAttr(&ctx, weightsElemBitSize), nullptr, getIntAttr(&ctx, channelOffset), getIntAttr(&ctx, 0));

    checkRelocateWeightsTableAttr(actualTransformations[0], expectedRelocateWeightsTableAttr);
    checkSubViewAttr(actualTransformations[1], offset, shape);
}

//
// Combinations
//

TEST_F(MLIR_ContentSetupTest, MoveSubViewReshapeBeforeAdd) {
    SmallVector<int64_t> baseContentShape = {4, 8, 3};
    auto baseContentAttrSetup = getContentSetup(ArrayRef(baseContentShape), getInt8Type(&ctx));

    // Add
    auto contentAttrSetup = baseContentAttrSetup.add(1);

    // SubView
    SmallVector<int64_t> expectedSubViewOffset = {0, 1, 1};
    SmallVector<int64_t> expectedSubViewShape = {1, 4, 2};

    contentAttrSetup = contentAttrSetup.subview(ShapeRef(expectedSubViewOffset), ShapeRef(expectedSubViewShape));

    // Reshape
    SmallVector<int64_t> expectedShape = {1, 2, 4};

    contentAttrSetup = contentAttrSetup.reshape(ShapeRef(expectedShape));

    // check
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 3);

    checkSubViewAttr(actualTransformations[0], expectedSubViewOffset, expectedSubViewShape);
    checkReshapeAttr(actualTransformations[1], expectedShape);
    checkAddAttr(actualTransformations[2], 1);
}

// MoveRelocateWeightsTableIntoFuse

TEST_F(MLIR_ContentSetupTest, MoveRelocateWeightsTableIntoFuse) {
    auto tensorType = getInt8Type(&ctx);
    auto fusedType = mlir::RankedTensorType::get({4}, tensorType);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{4}, tensorType);

    // Fuse
    SmallVector<uint8_t> data = {1, 1, 1, 1};
    auto weightsType = mlir::RankedTensorType::get(ArrayRef<int64_t>{4}, tensorType);
    auto weightsAttr = Const::createConstContent(weightsType, ArrayRef(data));
    auto weights = Const::ContentAttr::get(weightsAttr);
    auto weightsTable = Const::ContentAttr::get(weightsAttr);
    auto contentAttrSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, Const::ContentAttr{}, {});

    SmallVector<uint32_t> weightsPtr = {100, 200, 300, 400};
    auto sparsityPtr = 16777215;
    SmallVector<int64_t> offsets = {0, 2, 4, 6};
    auto weightsTableSize = 128;
    auto weightsElemBitSize = 16;
    auto channelOffset = 0;

    // RelocateWT
    contentAttrSetup =
            contentAttrSetup.relocateWeightsTablePointers(weightsPtr, sparsityPtr, ShapeRef(offsets), weightsTableSize,
                                                          weightsElemBitSize, nullptr, channelOffset, 0);

    // Fuse + RelocateWT should be fused into one Fuse transformation
    auto actualTransformations = contentAttrSetup.getTransformations();
    ASSERT_TRUE(actualTransformations.size() == 1);
    checkFuseAttr(actualTransformations[0]);

    // The resulting Fuses' weightsTable should have one transformation
    auto actualFuseAttr = mlir::cast<Const::FuseWeightsAttr>(actualTransformations[0]);
    auto actualWT = actualFuseAttr.getWeightsTable();
    auto wtTransformations = actualWT.getTransformations();
    ASSERT_TRUE(wtTransformations.size() == 1);

    auto expectedRelocateWeightsTableAttr = Const::RelocateWeightsTableAttr::get(
            getIntArrayAttr(&ctx, ArrayRef(weightsPtr)), getIntAttr(&ctx, sparsityPtr),
            getIntArrayAttr(&ctx, ArrayRef(offsets)), getIntAttr(&ctx, weightsTableSize),
            getIntAttr(&ctx, weightsElemBitSize), nullptr, getIntAttr(&ctx, channelOffset), getIntAttr(&ctx, 0));

    checkRelocateWeightsTableAttr(wtTransformations[0], expectedRelocateWeightsTableAttr);
}

// MoveSubViewIntoFuse

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseSplitInHalf) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});
    auto weights = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 2, 1, 1}, tensorType,
                                           SmallVector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto fusedType = mlir::RankedTensorType::get({1, 1, 1, 12}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});

    auto fusedLowerHalfSetup = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 0}, ShapeRef{1, 1, 1, 6});

    auto actualTransformationsLowerHalf = fusedLowerHalfSetup.getTransformations();
    EXPECT_EQ(actualTransformationsLowerHalf.size(), 1);
    checkFuseAttrAfterSubView(actualTransformationsLowerHalf[0], true, false, {}, {}, false, {}, {}, {}, true, true,
                              {0, 0, 0, 0}, {1, 2, 1, 1}, false, {}, {}, {});

    baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);
    fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});
    auto fusedUpperHalfSetup = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 6}, ShapeRef{1, 1, 1, 6});
    auto actualTransformationsUpperHalf = fusedUpperHalfSetup.getTransformations();
    EXPECT_EQ(actualTransformationsUpperHalf.size(), 1);
    checkFuseAttrAfterSubView(actualTransformationsUpperHalf[0], false, false, {}, {}, false, {}, {}, {}, true, true,
                              {1, 0, 0, 0}, {3, 2, 1, 1}, false, {}, {}, {});
}

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseUnalignedViewIntoWeights) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});
    auto weights = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 2}, tensorType,
                                           SmallVector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto fusedType = mlir::RankedTensorType::get({1, 1, 1, 12}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});

    auto fusedWithSubView = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 7}, ShapeRef{1, 1, 1, 3});

    auto actualTransformations = fusedWithSubView.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);
    checkFuseAttrAfterSubView(actualTransformations[0], false, false, {}, {}, false, {}, {}, {}, true, true,
                              {1, 0, 0, 0}, {2, 1, 1, 2}, true, {1, 1, 1, 4}, {0, 0, 0, 1}, {1, 1, 1, 3});
}

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseUnalignedViewIntoUnslicableWeights) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});
    auto weights = getContentAttr<uint8_t>(ArrayRef<int64_t>{2, 1, 2, 2}, tensorType,
                                           SmallVector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto fusedType = mlir::RankedTensorType::get({1, 1, 1, 12}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});

    auto fusedWithSubView = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 7}, ShapeRef{1, 1, 1, 3});
    auto actualTransformations = fusedWithSubView.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);
    checkFuseAttrAfterSubView(actualTransformations[0], false, false, {}, {}, false, {}, {}, {}, true, false, {}, {},
                              true, {1, 1, 1, 8}, {0, 0, 0, 3}, {1, 1, 1, 3});
}

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseSubByteWeights) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});

    const auto baseWeightsType = mlir::RankedTensorType::get({4, 1, 1, 4}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseWeightsAttr = Const::createConstContent(baseWeightsType, ArrayRef(vals));

    Const::ContentSetup baseWeightsContentAttrSetup(baseWeightsType);
    auto weights = Const::ContentAttr::get(baseWeightsAttr,
                                           baseWeightsContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4)));

    auto fusedType = mlir::RankedTensorType::get({1, 1, 1, 12}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});

    auto fusedWithSubView = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 8}, ShapeRef{1, 1, 1, 4});
    auto actualTransformations = fusedWithSubView.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    auto fuse = mlir::dyn_cast<Const::FuseWeightsAttr>(actualTransformations[0]);
    ASSERT_TRUE(fuse != nullptr);

    auto actualWeights = fuse.getWeights();
    ASSERT_TRUE(actualWeights != nullptr);
    auto actualWeightTransformations = actualWeights.getTransformations();
    EXPECT_EQ(actualWeightTransformations.size(), 2);
    checkSubViewAttr(actualWeightTransformations[0], {2, 0, 0, 0}, {2, 1, 1, 4});
}

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseDontMoveSparse) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});
    auto weights = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 2, 1, 1}, tensorType,
                                           SmallVector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto sparsity = getContentAttr<uint8_t>(ArrayRef<int64_t>{1, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1});
    auto fusedType = mlir::RankedTensorType::get({1, 1, 1, 13}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, sparsity, {});

    auto fusedConstantWithSubView = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 0}, ShapeRef{1, 1, 1, 7});

    auto actualTransformations = fusedConstantWithSubView.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);
    auto fuse = mlir::dyn_cast<Const::FuseWeightsAttr>(actualTransformations[0]);
    EXPECT_NE(fuse, nullptr);
    checkSubViewAttr(actualTransformations[1], {0, 0, 0, 0}, {1, 1, 1, 7});
}

TEST_F(MLIR_ContentSetupTest, MoveSubViewIntoFuseDontMoveUnsupportedShape) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto weightsTable =
            getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 1, 1, 1});
    auto weights = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 2, 1, 1}, tensorType,
                                           SmallVector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto fusedType = mlir::RankedTensorType::get({6, 1, 1, 2}, tensorType);
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, weightsTable, weights, {}, {});

    auto fusedConstantWithSubView = fusedConstantSetup.subview(ShapeRef{0, 0, 0, 0}, ShapeRef{1, 1, 1, 7});

    auto actualTransformations = fusedConstantWithSubView.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 2);
    auto fuse = mlir::dyn_cast<Const::FuseWeightsAttr>(actualTransformations[0]);
    EXPECT_NE(fuse, nullptr);
    checkSubViewAttr(actualTransformations[1], {0, 0, 0, 0}, {1, 1, 1, 7});
}

TEST_F(MLIR_ContentSetupTest, FuseConstants) {
    auto tensorType = getInt8Type(&ctx);

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{1}, tensorType);

    auto cst1 = getContentAttr<uint8_t>(ArrayRef<int64_t>{4, 1, 1, 1}, tensorType, SmallVector<uint8_t>{1, 2, 3, 4});
    auto cst2 = getContentAttr<uint8_t>(ArrayRef<int64_t>{2, 1, 1, 1}, tensorType, SmallVector<uint8_t>{5, 6});

    auto fusedType = mlir::RankedTensorType::get({6, 1, 1, 1}, tensorType);
    std::vector<Const::ContentAttr> constants{cst1, cst2};
    auto fusedConstantSetup = baseContentAttrSetup.fuse(fusedType, constants);

    auto actualTransformations = fusedConstantSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);
    auto fuse = mlir::dyn_cast<Const::FuseAttr>(actualTransformations[0]);
    EXPECT_NE(fuse, nullptr);
}

TEST_F(MLIR_ContentSetupTest, FuseReorderAndMemPermute) {
    const int64_t IC = 4;
    const int64_t IH = 6;
    const int64_t IW = 8;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::HWC);
    contentAttrSetup = contentAttrSetup.memPermute(DimsOrder::WHC, DimsOrder::HCW);

    // check final mem permute
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkMemPermuteAttr(actualTransformations[0], DimsOrder::WHC, DimsOrder::WHC);
}

TEST_F(MLIR_ContentSetupTest, FuseMemPermuteAndReorder) {
    const int64_t IC = 4;
    const int64_t IH = 6;
    const int64_t IW = 8;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.memPermute(DimsOrder::WHC, DimsOrder::WHC);
    contentAttrSetup = contentAttrSetup.reorder(DimsOrder::HWC);

    // check final mem permute
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkMemPermuteAttr(actualTransformations[0], DimsOrder::HWC, DimsOrder::HWC);
}

TEST_F(MLIR_ContentSetupTest, FoldReorder) {
    const int64_t IC = 4;
    const int64_t IH = 6;
    const int64_t IW = 8;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::CHW);

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 0);
}

TEST_F(MLIR_ContentSetupTest, DoNotFoldTrivialReorder) {
    const int64_t IC = 2;
    const int64_t IH = 1;
    const int64_t IW = 1;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::HWC);

    // we cand fold reorder since type is changed
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkReorderAttr(actualTransformations[0], DimsOrder::HWC);
}

TEST_F(MLIR_ContentSetupTest, FoldMemPerm) {
    const int64_t IC = 4;
    const int64_t IH = 6;
    const int64_t IW = 8;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.memPermute(DimsOrder::CHW, DimsOrder::CHW);

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 0);
}

TEST_F(MLIR_ContentSetupTest, DoNotFoldTrivialMemPerm) {
    const int64_t IC = 2;
    const int64_t IH = 1;
    const int64_t IW = 1;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.memPermute(DimsOrder::HWC, DimsOrder::HWC);

    // we cand fold mem permute since type is changed
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkMemPermuteAttr(actualTransformations[0], DimsOrder::HWC, DimsOrder::HWC);
}

TEST_F(MLIR_ContentSetupTest, FoldTranspose) {
    const int64_t IC = 4;
    const int64_t IH = 6;
    const int64_t IW = 8;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.transpose(DimsOrder::CHW);

    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 0);
}

TEST_F(MLIR_ContentSetupTest, DoNotFoldTrivialTranspose) {
    const int64_t IC = 2;
    const int64_t IH = 1;
    const int64_t IW = 1;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH, IW}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.transpose(DimsOrder::HWC);

    // we cannot fold transpose since type is changed
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkTransposeAttr(actualTransformations[0], DimsOrder::HWC);
}

TEST_F(MLIR_ContentSetupTest, DoNotFoldTypePreservingTranspose) {
    const int64_t IC = 2;
    const int64_t IH = 2;

    auto baseContentAttrSetup = getContentSetup(ArrayRef<int64_t>{IC, IH}, getInt8Type(&ctx));

    auto contentAttrSetup = baseContentAttrSetup.transpose(DimsOrder::CN);

    // we cannot fold transpose since the permutation is not identity
    auto actualTransformations = contentAttrSetup.getTransformations();
    EXPECT_EQ(actualTransformations.size(), 1);

    checkTransposeAttr(actualTransformations[0], DimsOrder::CN);
}
