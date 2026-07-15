//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>

using namespace vpux;

namespace {

std::vector<DimsOrder> getOrders() {
    return std::vector<DimsOrder>{vpux::DimsOrder::C,    vpux::DimsOrder::NC,   vpux::DimsOrder::CHW,
                                  vpux::DimsOrder::HWC,  vpux::DimsOrder::HCW,  vpux::DimsOrder::NCHW,
                                  vpux::DimsOrder::NHWC, vpux::DimsOrder::NHCW, vpux::DimsOrder::NCDHW,
                                  vpux::DimsOrder::NDHWC};
}

std::vector<std::pair<DimsOrder, size_t>> getOrders2Dims() {
    return std::vector<std::pair<DimsOrder, size_t>>{
            std::make_pair(vpux::DimsOrder::C, 1u),     std::make_pair(vpux::DimsOrder::NC, 2u),
            std::make_pair(vpux::DimsOrder::CHW, 3u),   std::make_pair(vpux::DimsOrder::HWC, 3u),
            std::make_pair(vpux::DimsOrder::HCW, 3u),   std::make_pair(vpux::DimsOrder::NCHW, 4u),
            std::make_pair(vpux::DimsOrder::NHWC, 4u),  std::make_pair(vpux::DimsOrder::NHCW, 4u),
            std::make_pair(vpux::DimsOrder::NCDHW, 5u), std::make_pair(vpux::DimsOrder::NDHWC, 5u)};
}

std::vector<std::pair<std::vector<Dim>, DimsOrder>> getPerm2Order() {
    return std::vector<std::pair<std::vector<Dim>, DimsOrder>>{
            std::make_pair(std::vector<Dim>({Dim(0)}), DimsOrder::C),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1)}), DimsOrder::NC),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2)}), DimsOrder::CHW),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(2), Dim(0)}), DimsOrder::HWC),
            std::make_pair(std::vector<Dim>({Dim(1), Dim(0), Dim(2)}), DimsOrder::HCW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2), Dim(3)}), DimsOrder::NCHW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(3), Dim(1)}), DimsOrder::NHWC),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(1), Dim(3)}), DimsOrder::NHCW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(1), Dim(2), Dim(3), Dim(4)}), DimsOrder::NCDHW),
            std::make_pair(std::vector<Dim>({Dim(0), Dim(2), Dim(3), Dim(4), Dim(1)}), DimsOrder::NDHWC)};
}
std::vector<std::pair<DimsOrder::StorageType, DimsOrder>> getCode2Order() {
    return std::vector<std::pair<DimsOrder::StorageType, DimsOrder>>{
            std::make_pair(0x1, DimsOrder::C),         std::make_pair(0x12, DimsOrder::NC),
            std::make_pair(0x123, DimsOrder::CHW),     std::make_pair(0x231, DimsOrder::HWC),
            std::make_pair(0x213, DimsOrder::HCW),     std::make_pair(0x1234, DimsOrder::NCHW),
            std::make_pair(0x1342, DimsOrder::NHWC),   std::make_pair(0x1324, DimsOrder::NHCW),
            std::make_pair(0x12345, DimsOrder::NCDHW), std::make_pair(0x13452, DimsOrder::NDHWC)};
}

std::vector<std::pair<DimsOrder, StringRef>> getOrders2Name() {
    return std::vector<std::pair<DimsOrder, StringRef>>{
            std::make_pair(vpux::DimsOrder(), "SCALAR"),    std::make_pair(vpux::DimsOrder::C, "C"),
            std::make_pair(vpux::DimsOrder::NC, "NC"),      std::make_pair(vpux::DimsOrder::CHW, "CHW"),
            std::make_pair(vpux::DimsOrder::HWC, "HWC"),    std::make_pair(vpux::DimsOrder::HCW, "HCW"),
            std::make_pair(vpux::DimsOrder::NCHW, "NCHW"),  std::make_pair(vpux::DimsOrder::NHWC, "NHWC"),
            std::make_pair(vpux::DimsOrder::NHCW, "NHCW"),  std::make_pair(vpux::DimsOrder::NCDHW, "NCDHW"),
            std::make_pair(vpux::DimsOrder::NDHWC, "NDHWC")};
}

using ShapeType = SmallVector<int64_t>;
using MemStridesType = SmallVector<int64_t>;
using ExpectedDimsOrderType = DimsOrder;
using ValueDimsOrderType = DimsOrder;
using ResType = bool;

std::vector<ShapeType> getFromType() {
    return {
            {2}, {2, 3}, {1, 3, 6}, {5, 4, 3, 2}, {5, 1, 2, 3, 5},
    };
}

std::vector<std::tuple<ValueDimsOrderType, ShapeType>> getFromMemRefTypeWithLayout() {
    return {
            {DimsOrder::CHW, {8, 4, 2}},     {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 4, 3, 2}},
            {DimsOrder::NHCW, {5, 1, 2, 3}}, {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 1, 2, 3}},
    };
}

std::vector<std::tuple<ValueDimsOrderType, ShapeType>> getFromTensorType() {
    return {
            {DimsOrder::CHW, {8, 4, 2}},     {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 4, 3, 2}},
            {DimsOrder::NHCW, {5, 1, 2, 3}}, {DimsOrder::HCW, {1, 3, 6}}, {DimsOrder::NHCW, {5, 1, 2, 3}},
    };
}

std::vector<std::tuple<ExpectedDimsOrderType, ValueDimsOrderType, ShapeType, MemStridesType, ResType>>
getIsCompatibleLayout() {
    return {
            {DimsOrder::CHW, DimsOrder::CHW, {1, 2, 4}, {8, 4, 1}, true},

            {DimsOrder::HCW, DimsOrder::CHW, {1, 1, 6}, {6, 6, 1}, true},
            {DimsOrder::HCW, DimsOrder::CHW, {2, 2, 6}, {12, 6, 1}, false},

            {DimsOrder::NCHW, DimsOrder::NHCW, {5, 2, 1, 3}, {6, 6, 3, 1}, false},
            {DimsOrder::NCHW, DimsOrder::HCW, {1, 1, 1}, {1, 1, 1}, false},  // type.getRank() != numDims()
    };
}

}  // namespace

class MLIR_DimsOrderTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_DimsOrderTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<VPUIP::VPUIPDialect>();
    }
};

TEST_F(MLIR_DimsOrderTest, ValidateCodeTest) {
    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_NO_THROW(DimsOrder::validateCode(order.code()));
    });

    // check double usage dimension
    EXPECT_ANY_THROW(DimsOrder::validateCode(0x11));
    // check negative code
    EXPECT_ANY_THROW(DimsOrder::validateCode(-10));
}

TEST_F(MLIR_DimsOrderTest, ValidateNumDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST_F(MLIR_DimsOrderTest, ValidatePermutationTest) {
    std::vector<DimArr> validPermutation{
            {Dim(0)},

            {Dim(0), Dim(1)},
            {Dim(1), Dim(0)},

            {Dim(0), Dim(1), Dim(2)},
            {Dim(0), Dim(2), Dim(1)},
            {Dim(1), Dim(0), Dim(2)},
            {Dim(1), Dim(2), Dim(0)},
            {Dim(2), Dim(0), Dim(1)},
            {Dim(2), Dim(1), Dim(0)},

            {Dim(0), Dim(1), Dim(2), Dim(3)},
            {Dim(3), Dim(2), Dim(1), Dim(0)},
            {Dim(0), Dim(3), Dim(1), Dim(2)},
    };
    std::for_each(validPermutation.begin(), validPermutation.end(), [](const DimArr& perm) {
        EXPECT_NO_THROW(DimsOrder::validatePermutation(perm));
    });
    std::vector<DimArr> invalidPermutation{
            {Dim(1)},
            {Dim(2)},

            {Dim(1), Dim(2)},
            {Dim(2), Dim(3)},
            {Dim(1), Dim(2)},

            {Dim(0), Dim(1), Dim(1)},
            {Dim(1), Dim(2), Dim(3)},
            {Dim(0), Dim(0), Dim(0)},
    };
    std::for_each(invalidPermutation.begin(), invalidPermutation.end(), [](const DimArr& perm) {
        EXPECT_ANY_THROW(DimsOrder::validatePermutation(perm));
    });
}

TEST_F(MLIR_DimsOrderTest, getCodeFromNumDimsTest) {
    std::vector<DimsOrder> defOrders{DimsOrder::C, DimsOrder::NC, DimsOrder::CHW, DimsOrder::NCHW, DimsOrder::NCDHW};

    for (size_t i = 0; i < defOrders.size(); ++i) {
        EXPECT_EQ(DimsOrder::getCodeFromNumDims(i + 1), defOrders[i].code());
    }
}

TEST_F(MLIR_DimsOrderTest, getCodeFromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::getCodeFromPermutation(perm2order.first), perm2order.second.code());
                  });
}

TEST_F(MLIR_DimsOrderTest, fromCodeTest) {
    auto codes2orders = getCode2Order();

    std::for_each(codes2orders.begin(), codes2orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(DimsOrder::fromCode(code2order.first), code2order.second);
                  });
}

TEST_F(MLIR_DimsOrderTest, fromNumDimsTest) {
    std::vector<std::pair<size_t, DimsOrder>> numDims2DimsOrderTable{
            std::make_pair(1, DimsOrder::C), std::make_pair(2, DimsOrder::NC), std::make_pair(3, DimsOrder::CHW),
            std::make_pair(4, DimsOrder::NCHW), std::make_pair(5, DimsOrder::NCDHW)};

    std::for_each(numDims2DimsOrderTable.begin(), numDims2DimsOrderTable.end(),
                  [](const std::pair<size_t, DimsOrder>& numDims2order) {
                      EXPECT_EQ(DimsOrder::fromNumDims(numDims2order.first), numDims2order.second);
                  });
}

TEST_F(MLIR_DimsOrderTest, FromPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(perms2orders.begin(), perms2orders.end(),
                  [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                      EXPECT_EQ(DimsOrder::fromPermutation(perm2order.first), perm2order.second);
                  });
}

TEST_F(MLIR_DimsOrderTest, codeTest) {
    DimsOrder dim;
    EXPECT_EQ(dim.code(), 0);

    auto codes2Orders = getCode2Order();

    std::for_each(codes2Orders.begin(), codes2Orders.end(),
                  [](const std::pair<DimsOrder::StorageType, DimsOrder>& code2order) {
                      EXPECT_EQ(code2order.first, code2order.second.code());
                  });
}

TEST_F(MLIR_DimsOrderTest, emptyTest) {
    DimsOrder dim;
    EXPECT_TRUE(dim.empty());

    auto orders = getOrders();

    std::for_each(orders.begin(), orders.end(), [](const DimsOrder& order) {
        EXPECT_FALSE(order.empty());
    });
}

TEST_F(MLIR_DimsOrderTest, numDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}

TEST_F(MLIR_DimsOrderTest, hasDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_TRUE(order2dim.first.hasDim(Dim(i)));
            } else {
                EXPECT_FALSE(order2dim.first.hasDim(Dim(i)));
            }
        }
    });
}

TEST_F(MLIR_DimsOrderTest, dimPosTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_EQ(order2dim.first.dimPos(order2dim.first.dimAt(i)), i);
            } else {
                EXPECT_ANY_THROW(order2dim.first.dimPos(Dim(i)));
            }
        }
    });
}

TEST_F(MLIR_DimsOrderTest, dimPosTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(3)), 3);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(2)), 2);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(1)), 1);
    EXPECT_EQ(DimsOrder::NCHW.dimPos(Dim(0)), 0);

    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(1)), 3);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(3)), 2);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(2)), 1);
    EXPECT_EQ(DimsOrder::NHWC.dimPos(Dim(0)), 0);
}

TEST_F(MLIR_DimsOrderTest, dimAtTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        for (size_t i = 0; i < MAX_NUM_DIMS; ++i) {
            if (i < order2dim.second) {
                EXPECT_NO_THROW(order2dim.first.dimAt(i));
            } else {
                EXPECT_ANY_THROW(order2dim.first.dimAt(i));
            }
        }
    });
}

TEST_F(MLIR_DimsOrderTest, dimAtTest_4D) {
    EXPECT_EQ(DimsOrder::NCHW.dimAt(0), Dim(0));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(1), Dim(1));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(2), Dim(2));
    EXPECT_EQ(DimsOrder::NCHW.dimAt(3), Dim(3));

    EXPECT_EQ(DimsOrder::NHWC.dimAt(0), Dim(0));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(1), Dim(2));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(2), Dim(3));
    EXPECT_EQ(DimsOrder::NHWC.dimAt(3), Dim(1));
}

TEST_F(MLIR_DimsOrderTest, toPermutationTest) {
    auto perms2orders = getPerm2Order();

    std::for_each(
            perms2orders.begin(), perms2orders.end(), [](const std::pair<std::vector<Dim>, DimsOrder>& perm2order) {
                EXPECT_EQ(DimArr(perm2order.first.begin(), perm2order.first.end()), perm2order.second.toPermutation());
            });
}

TEST_F(MLIR_DimsOrderTest, MemDim_4D) {
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(0)), MemDim(0));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(1)), MemDim(1));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(2)), MemDim(2));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(3)), MemDim(3));

    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(0)), MemDim(0));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(1)), MemDim(3));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(2)), MemDim(1));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(3)), MemDim(2));
}

TEST_F(MLIR_DimsOrderTest, TryToSetIncorrectDimIndx) {
    EXPECT_ANY_THROW(Dim(-1));
    EXPECT_ANY_THROW(Dim(-123456));
}

TEST_F(MLIR_DimsOrderTest, getCanonicalName) {
    auto orders2name = getOrders2Name();

    std::for_each(orders2name.begin(), orders2name.end(), [](const std::pair<DimsOrder, StringRef>& order2name) {
        const auto name = order2name.first.getCanonicalName();
        ASSERT_TRUE(!name.empty()) << order2name.second.data();
        ASSERT_EQ(name, order2name.second);
    });

    const auto nonDefault = DimsOrder::fromNumDims(7).getCanonicalName();
    ASSERT_TRUE(nonDefault.empty());
}

TEST_F(MLIR_DimsOrderTest, fromMemRefTypeSimpleTest) {
    const auto testData = getFromType();

    for (const auto& shape : testData) {
        const auto memRefType = mlir::MemRefType::get(shape, mlir::Float16Type::get(&ctx));

        const auto actualOrder = mlir::cast<vpux::NDTypeInterface>(memRefType).getDimsOrder();
        EXPECT_EQ(DimsOrder::fromNumDims(shape.size()), actualOrder);
    }
}

TEST_F(MLIR_DimsOrderTest, fromMemRefTypeWithLayoutTest) {
    const auto testData = getFromMemRefTypeWithLayout();

    for (const auto& testCase : testData) {
        DimsOrder expOrder;
        SmallVector<int64_t> shape;
        std::tie(expOrder, shape) = testCase;

        const auto memRefType =
                getMemRefType(ShapeRef(shape), mlir::Float16Type::get(&ctx), expOrder, /*memSpace=*/nullptr);

        const auto actualOrder = mlir::cast<vpux::NDTypeInterface>(memRefType).getDimsOrder();
        EXPECT_EQ(expOrder, actualOrder);
    }
}

TEST_F(MLIR_DimsOrderTest, fromTensorTypeTest) {
    const auto testData = getFromType();

    for (const auto& shape : testData) {
        const auto tensorType = mlir::RankedTensorType::get(shape, mlir::Float16Type::get(&ctx));

        const auto actualOrder = mlir::cast<vpux::NDTypeInterface>(tensorType).getDimsOrder();
        EXPECT_EQ(DimsOrder::fromNumDims(shape.size()), actualOrder);
    }
}

TEST_F(MLIR_DimsOrderTest, fromTensorTypeTest_WithMaps) {
    const auto testData = getFromTensorType();

    for (const auto& testCase : testData) {
        DimsOrder expOrder;
        SmallVector<int64_t> shape{};

        std::tie(expOrder, shape) = testCase;

        const auto tensorType = getTensorType(ShapeRef(shape), mlir::Float16Type::get(&ctx), expOrder, nullptr);

        const auto actualOrder = mlir::cast<vpux::NDTypeInterface>(tensorType).getDimsOrder();
        EXPECT_EQ(expOrder, actualOrder);
    }
}

TEST_F(MLIR_DimsOrderTest, isCompatibleLayoutTest) {
    const auto testData = getIsCompatibleLayout();

    for (const auto& testCase : testData) {
        DimsOrder expOrder;
        DimsOrder originOrder;
        SmallVector<int64_t> shape;
        SmallVector<int64_t> memStrides;
        ResType isCompatible;
        std::tie(expOrder, originOrder, shape, memStrides, isCompatible) = testCase;

        SmallVector<int64_t> elemStrides(memStrides.size());
        for (auto i : irange(memStrides.size())) {
            elemStrides[i] = memStrides[checked_cast<size_t>(originOrder.toMemDim(Dim(i)).ind())];
        }

        const auto layout = vpux::MemRefAttr::get(mlir::AffineMapAttr::get(originOrder.toAffineMap(&ctx)),
                                                  getIntArrayAttr(&ctx, elemStrides),
                                                  /*allocSize=*/nullptr, &ctx);

        const auto memRefType = mlir::MemRefType::get(shape, mlir::Float16Type::get(&ctx),
                                                      mlir::cast<mlir::MemRefLayoutAttrInterface>(layout));

        EXPECT_EQ(expOrder.isCompatibleLayout(memRefType), isCompatible) << printToString(
                "expOrder = {0} memRefType = {1} isCompatible = {2}", expOrder, memRefType, isCompatible);
    }
}

TEST_F(MLIR_DimsOrderTest, copyMoveSemanticsPreserveState) {
    const auto base = DimsOrder::NHCW;
    const auto expectedCode = base.code();
    const auto expectedInvCode = base.invertedCode();
    const auto expectedNumDims = base.numDims();
    const auto expectedPerm = base.toPermutation();

    const DimsOrder copyCtor(base);
    EXPECT_EQ(copyCtor.code(), expectedCode);
    EXPECT_EQ(copyCtor.invertedCode(), expectedInvCode);
    EXPECT_EQ(copyCtor.numDims(), expectedNumDims);
    EXPECT_EQ(copyCtor.toPermutation(), expectedPerm);

    DimsOrder copyAssign = DimsOrder::CHW;
    copyAssign = base;
    EXPECT_EQ(copyAssign.code(), expectedCode);
    EXPECT_EQ(copyAssign.invertedCode(), expectedInvCode);
    EXPECT_EQ(copyAssign.numDims(), expectedNumDims);
    EXPECT_EQ(copyAssign.toPermutation(), expectedPerm);

    DimsOrder moveCtorSource = base;
    DimsOrder moveCtor(std::move(moveCtorSource));
    EXPECT_EQ(moveCtor.code(), expectedCode);
    EXPECT_EQ(moveCtor.invertedCode(), expectedInvCode);
    EXPECT_EQ(moveCtor.numDims(), expectedNumDims);
    EXPECT_EQ(moveCtor.toPermutation(), expectedPerm);

    DimsOrder moveAssignSource = base;
    DimsOrder moveAssign = DimsOrder::CHW;
    moveAssign = std::move(moveAssignSource);
    EXPECT_EQ(moveAssign.code(), expectedCode);
    EXPECT_EQ(moveAssign.invertedCode(), expectedInvCode);
    EXPECT_EQ(moveAssign.numDims(), expectedNumDims);
    EXPECT_EQ(moveAssign.toPermutation(), expectedPerm);
}
