//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/batch.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

#include <gtest/gtest.h>
#include "common/utils.hpp"

using namespace vpux;

using DebatchCoeffDescriptionTest = MLIR_UnitBase;

TEST_F(DebatchCoeffDescriptionTest, ParamsWellFormed) {
    std::string str = "[1-2]";
    DebatchCoeffDescription v;
    ASSERT_NO_THROW(v = DebatchCoeffDescription::createFromString(str));
    ASSERT_NE(v.batchPositionIndex, DebatchCoeffDescription{}.batchPositionIndex);
    ASSERT_NE(v.desiredBatchValue, DebatchCoeffDescription{}.desiredBatchValue);
    ASSERT_EQ(v.batchPositionIndex, Dim{1});
    ASSERT_EQ(v.desiredBatchValue, 2);
}

TEST_F(DebatchCoeffDescriptionTest, ParamsFormatViolatedStructure) {
    std::string str = "1-2]";
    ASSERT_THROW(DebatchCoeffDescription::createFromString(str), std::exception);
    str = "[1-2";
    ASSERT_THROW(DebatchCoeffDescription::createFromString(str), std::exception);
    str = "[]";
    ASSERT_THROW(DebatchCoeffDescription::createFromString(str), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, ParamsFormatViolatedValues) {
    std::string str = "[a-2]";
    ASSERT_THROW(DebatchCoeffDescription::createFromString(str), std::exception);
    str = "[1-b]";
    ASSERT_THROW(DebatchCoeffDescription::createFromString(str), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, ApplyShapeSuccess) {
    std::string str = "[0-2]";
    Shape shape{{4, 4, 4}};
    for (size_t i = 0; i < shape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.apply(shape);
        ASSERT_EQ(resultShape[Dim{i}], v.desiredBatchValue);
        for (size_t j = 0; j < shape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], shape[Dim{j}]);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, ApplyDynamicShapeSuccess) {
    std::string str = "[0-3]";
    Shape shape{{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}};
    for (size_t i = 0; i < shape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.apply(shape);
        ASSERT_EQ(resultShape[Dim{i}], v.desiredBatchValue);
        for (size_t j = 0; j < shape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], shape[Dim{j}]);
            }
        }
    }

    DebatchCoeffDescription dynCoeff;
    dynCoeff.desiredBatchValue = mlir::ShapedType::kDynamic;
    for (size_t i = 0; i < shape.size(); i++) {
        dynCoeff.batchPositionIndex = Dim{i};
        Shape resultShape = dynCoeff.apply(shape);
        ASSERT_EQ(resultShape[Dim{i}], mlir::ShapedType::kDynamic);
        for (size_t j = 0; j < shape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], shape[Dim{j}]);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, ApplyShapeFail) {
    std::string str = "[0-3]";
    Shape shape{{4, 4, 4}};
    auto v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.apply(shape), std::exception);

    str = "[1-3]";
    v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.apply(shape), std::exception);

    str = "[10-3]";
    v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.apply(shape), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromShapeDowncastSuccess) {
    std::string str = "[0-2]";
    Shape fromShape{{4, 4, 4}};
    Shape toShape{{6, 6, 6}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.applyProportionFromShape(fromShape, toShape);
        ASSERT_EQ(resultShape[Dim{i}], 3);
        for (size_t j = 0; j < fromShape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], toShape[Dim{j}]);
            }
        }
    }
}
TEST_F(DebatchCoeffDescriptionTest, applyProportionFromDynamicShapeDowncastSuccess) {
    std::string str = "[0-1]";
    Shape fromShape{{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}};
    Shape toShape{{3, 1, 2}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.applyProportionFromShape(fromShape, toShape);
        ASSERT_EQ(resultShape[Dim{i}], 1);
        for (size_t j = 0; j < fromShape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], toShape[Dim{j}]);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromShapeUpcastSuccess) {
    std::string str = "[0-9]";
    Shape fromShape{{3, 3, 3}};
    Shape toShape{{6, 6, 6}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.applyProportionFromShape(fromShape, toShape);
        ASSERT_EQ(resultShape[Dim{i}], 18);
        for (size_t j = 0; j < fromShape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], toShape[Dim{j}]);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromShapeAvoidNullInResult) {
    std::string str = "[0-2]";
    Shape fromShape{{6, 6, 6}};
    Shape toShape{{2, 2, 2}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.applyProportionFromShape(fromShape, toShape);
        ASSERT_EQ(resultShape[Dim{i}], 2);
        for (size_t j = 0; j < fromShape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], toShape[Dim{j}]);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromDynamicShapeSuccess) {
    std::string str = "[0-3]";
    Shape fromShape{{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}};
    Shape toShape{{6, 6, 6}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        str[1] = std::to_string(i)[0];
        auto v = DebatchCoeffDescription::createFromString(str);
        Shape resultShape = v.applyProportionFromShape(fromShape, toShape);
        ASSERT_EQ(resultShape[Dim{i}], v.desiredBatchValue);
        for (size_t j = 0; j < fromShape.size(); j++) {
            if (i != j) {
                ASSERT_EQ(resultShape[Dim{j}], 6);
            }
        }
    }
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromShapeDowncastFail) {
    std::string str = "[0-3]";
    Shape fromShapeIndivisible{{4, 4, 4}};
    Shape toShape{{4, 4, 4}};
    auto v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.applyProportionFromShape(fromShapeIndivisible, toShape), std::exception);

    Shape fromShapeDivisible{{6, 6, 6}};
    Shape toShapeUndivisible{{3, 3, 3}};
    v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.applyProportionFromShape(fromShapeDivisible, toShapeUndivisible), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, applyProportionFromShapeUpcastFail) {
    std::string str = "[0-5]";
    Shape fromShape{{4, 4, 4}};
    Shape toShape{{4, 4, 4}};
    auto v = DebatchCoeffDescription::createFromString(str);
    ASSERT_THROW(v.applyProportionFromShape(fromShape, toShape), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, createFromShapesSuccess) {
    Shape fromShape{{4, 4, 4}};
    for (size_t i = 0; i < fromShape.size(); i++) {
        Shape toShape(fromShape);
        toShape[Dim{i}] = 1;
        auto v = DebatchCoeffDescription::createFromShapes(fromShape, toShape);
        Shape resultShape = v.apply(fromShape);
        ASSERT_EQ(resultShape, toShape);

        toShape[Dim{i}] = 8;
        v = DebatchCoeffDescription::createFromShapes(fromShape, toShape);
        resultShape = v.apply(fromShape);
        ASSERT_EQ(resultShape, toShape);
    }
}

TEST_F(DebatchCoeffDescriptionTest, createFromShapesFail) {
    Shape fromShape{{4, 4, 4}};
    Shape toSmallShape{{4, 1}};
    ASSERT_THROW(DebatchCoeffDescription::createFromShapes(fromShape, toSmallShape), std::exception);

    Shape toBigShape{{4, 6, 6, 7, 8, 9}};
    ASSERT_THROW(DebatchCoeffDescription::createFromShapes(fromShape, toBigShape), std::exception);

    Shape toImpossibleShape{{3, 4, 4}};
    ASSERT_THROW(DebatchCoeffDescription::createFromShapes(fromShape, toImpossibleShape), std::exception);
}

TEST_F(DebatchCoeffDescriptionTest, createFromShapesBatchNotFound) {
    Shape fromShape{{4, 4, 4}};
    Shape toShape{{4, 4, 4}};
    auto coeff = DebatchCoeffDescription::createFromShapes(fromShape, toShape);
    auto expectedCoeff = DebatchCoeffDescription{Dim{0}, toShape[Dim{0}]};
    ASSERT_EQ(coeff.batchPositionIndex, expectedCoeff.batchPositionIndex);
    ASSERT_EQ(coeff.desiredBatchValue, expectedCoeff.desiredBatchValue);
}

using DebatchCoefficientsTest = MLIR_UnitBase;
TEST_F(DebatchCoefficientsTest, ParamsWellFormed) {
    std::string str = "";
    std::optional<DebatchCoefficients> c;
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));
    ASSERT_EQ(c.has_value(), false);

    str = "1:[1-1],2:[2-2],3:[3-3]";
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));
    ASSERT_EQ(c->size(), 3);
    ASSERT_EQ(c->to_string(true), str);

    str = "[1-1],[2-2],[3-3]";
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));
    ASSERT_EQ(c->size(), 3);
    ASSERT_EQ(c->to_string(), str);

    str = "[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-4097],[0-4096],[0-4096],[0-4096],[0-1],[0-11008],["
          "0-11008],[0-11008],[0-11008],[0-4096],[0-4096],[0-1],[0-4096],[0-4096],[0-4096],[0-4096],[0-1],[0-1],[0-1],["
          "0-1],[0-1],[0-1],";
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));
    ASSERT_EQ(c->size(), 31);
    ASSERT_EQ(c->to_string(), str.substr(0, str.size() - 1 /*exclude the last comma*/));
}

TEST_F(DebatchCoefficientsTest, ParamsWellNamedUnnamedMalformed) {
    std::string str{"1:[1-1],[2-2],3:[3-3]"};
    ASSERT_THROW(DebatchCoefficients::create(str), std::exception);

    str = "1:[1-1],[2-2],rte[3-3]";
    ASSERT_THROW(DebatchCoefficients::create(str), std::exception);
}

TEST_F(DebatchCoefficientsTest, IndexedAccess) {
    std::string str{"[1-1],[2-2],[2-3]"};
    std::optional<DebatchCoefficients> c;
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));

    std::vector<Shape> shapes{{{3, 3, 3}, {4, 4, 4}, {6, 6, 6}}};
    ASSERT_EQ(shapes.size(), c->size());
    for (size_t i = 0; i < shapes.size(); i++) {
        ASSERT_NO_THROW(c->apply(shapes[i], i));
        ASSERT_THROW(c->apply(shapes[i], "i"), std::exception);
    }
}

TEST_F(DebatchCoefficientsTest, IndexedAccessMultiParams) {
    std::string str{"[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],[0-4096],[0-4096],[0-4096],[0-4096],[0-1],["
                    "0-11008],[0-11008],[0-11008],[0-11008],[0-4096],[0-4096],[0-1],[0-4096],[0-4096],[0-4096],[0-4096]"
                    ",[0-1],[0-1],[0-1],[0-1],[0-1],[0-1],"};
    std::optional<DebatchCoefficients> c;
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));

    Shape shape{4096, 4096};
    Shape modifiedShape;
    ASSERT_NO_THROW(modifiedShape = c->apply(shape, 10));
    ASSERT_EQ(shape, modifiedShape);
}

TEST_F(DebatchCoefficientsTest, NamedAccess) {
    std::string str{"0:[1-1],1:[2-2],2:[2-3]"};
    std::optional<DebatchCoefficients> c;
    ASSERT_NO_THROW(c = DebatchCoefficients::create(str));

    std::vector<Shape> shapes{{{3, 3, 3}, {4, 4, 4}, {6, 6, 6}}};
    ASSERT_EQ(shapes.size(), c->size());
    for (size_t i = 0; i < shapes.size(); i++) {
        ASSERT_NO_THROW(c->apply(shapes[i], std::to_string(i)));
        ASSERT_THROW(c->apply(shapes[i], i), std::exception);
    }
}
