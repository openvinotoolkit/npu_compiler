#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, relu)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto relu = om.relu(input);
    auto reluOp = om.getSourceOp(relu);
    auto output = om.output(relu);

    ASSERT_EQ(relu->getShape(), mv::Shape({32, 32, 3}));
    ASSERT_EQ(reluOp->getOpType(), "Relu");
    ASSERT_EQ(reluOp->attrsCount(), 2);
    ASSERT_EQ(reluOp->inputSlots(), 1);
    ASSERT_EQ(reluOp->outputSlots(), 1);
    ASSERT_EQ(relu->attrsCount(), 5);
    //ASSERT_TRUE(reluOp->isExecutable());

}
