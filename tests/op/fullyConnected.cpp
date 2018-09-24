#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, fullyConnected)
{

    mv::OpModel om("testModel");
    auto input = om.input({8, 8, 16}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> weightsData = mv::utils::generateSequence<double>(input->getShape().totalSize() * 100u);
    auto weights1 = om.constant(weightsData, {input->getShape().totalSize(), 100}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto fullyConnected = om.fullyConnected(input, weights1);
    auto fullyConnectedOp = om.getSourceOp(fullyConnected);
    auto output = om.output(fullyConnected);

    ASSERT_EQ(output->getShape(), mv::Shape({1, 100}));
    ASSERT_EQ(fullyConnectedOp->getOpType(), mv::OpType::FullyConnected);
    ASSERT_EQ(fullyConnectedOp->attrsCount(), 7);
    ASSERT_EQ(fullyConnectedOp->inputSlots(), 2);
    ASSERT_EQ(fullyConnectedOp->outputSlots(), 1);
    ASSERT_TRUE(fullyConnectedOp->isExecutable());

}
