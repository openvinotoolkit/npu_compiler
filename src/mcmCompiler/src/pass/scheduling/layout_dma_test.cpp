#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"

namespace {

class LayoutDMATest : public ::testing::Test {
  protected:

    LayoutDMATest() {}

    bool isGraphfileTensor(mv::Tensor* t)
    {
        return t->hasAttr("allocators")
          && t->get<std::set<std::string>>("allocators").count("GraphFile");
    }

    template <typename F>
    void forEachGraphfileTensor(mv::OpModel& opModel, const F& func)
    {
        for (auto ti = opModel.tensorBegin(); ti != opModel.tensorEnd(); ++ti)
        {
            mv::Tensor* t = &*ti;
            if (!t->hasAttr("allocators") || !t->get<std::set<std::string>>("allocators").count("GraphFile"))
            {
                continue;
            }
            if (t->hasAttr("splitStrategy")
                && !t->hasAttr("weightTable")
                && !t->hasAttr("sparsityMap")
                && t->get<std::string>("splitStrategy") == "SplitOverK")
            {
                unsigned numClusters = opModel.getGlobalConfigParams()->get<int>("Number_of_Clusters");
                for (unsigned j = 0; j < numClusters; ++j)
                {
                    func(&t->getSubTensor(j), t);
                }
            }
            else
            {
                func(t, t);
            }
        }
    }

    void LoadModel(mv::CompilationUnit* unit) {
        mv::OpModel& om = unit->model();

        auto input0 = om.input("input#9", {56,56,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
        input0->setQuantParams({{128},{0.007843137718737125},{-1.0},{1.0}});

        std::vector<int64_t> filterData0 = mv::utils::generateSequence<int64_t>(3*3*3*64);
        auto filter0 = om.constantInt("conv#0_filter#1", filterData0,{3,3,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
        filter0->setQuantParams({{135},{0.0025439101736992598},{-0.3435550332069397},{0.3051420748233795}});

        auto conv0 = om.conv("conv#10", input0, filter0, {1, 1}, {1, 1, 1, 1}, 1, 1);
        conv0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t>(64);
        mv::Data::TensorIterator biasWeights0 = om.constantInt("conv#0_bias#2", biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
        biasWeights0->setQuantParams({{0},{1.9952236470999196e-05},{-inf_},{inf_}});
        auto bias_c0 = om.bias("", conv0, biasWeights0);
        bias_c0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        std::vector<int64_t> filterData1 = mv::utils::generateSequence<int64_t> (3*3*64*128);
        auto filter1 = om.constantInt("conv_1#3_filter#4", filterData1,{3,3,64,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
        filter1->setQuantParams({{125},{0.003295167814940214},{-0.41293057799339294},{0.4273372292518616}});
        auto conv1 = om.conv("conv_1#11", bias_c0, filter1, {1, 1}, {1, 1, 1, 1}, 1, 1);
        conv1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (128);
        auto biasWeights1 = om.constantInt("conv_1#3_bias#5", biasWeightsData1,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
        biasWeights1->setQuantParams({{0},{1.292222714255331e-05},{-inf_},{inf_}});
        auto bias_c1 = om.bias("", conv1, biasWeights1);
        bias_c1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        std::vector<int64_t> filterData2 = mv::utils::generateSequence<int64_t> (3*3*128*128);
        auto filter2 = om.constantInt("output#6_filter#7", filterData2,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
        filter2->setQuantParams({{118},{0.0037134578451514244},{-0.44002026319503784},{0.5069115161895752}});
        auto conv2 = om.conv("output#12", bias_c1, filter2, {1, 1}, {1, 1, 1, 1}, 1, 1);
        conv2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (128);
        auto biasWeights2 = om.constantInt("output#6_bias#8", biasWeightsData2,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
        biasWeights2->setQuantParams({{0},{1.4562579963239841e-05},{-inf_},{inf_}});
        auto bias_c2 = om.bias("", conv2, biasWeights2);
        bias_c2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

        om.output("", bias_c2);

        unit->loadCompilationDescriptor(mv::Target::ma3100);  // THB
        unit->loadTargetDescriptor(mv::Target::ma3100);  // THB
    }

    const double inf_ = std::numeric_limits<double>::infinity();
};

// Verify that all graphfile tensors are assigned sequential indices.
TEST_F(LayoutDMATest, smoke)
{
    mv::CompilationUnit unit{"conv"};
    LoadModel(&unit);
    unit.initialize();
    unit.run();

    std::vector<bool> idxs;

    forEachGraphfileTensor(unit.model(), [&](mv::Tensor* t, mv::Tensor* /* p */)
        {
            unsigned idx = t->get<unsigned>("graphFileIndex");

            if (idxs.size() <= idx)
            {
                idxs.resize(idx + 1, false);
            }

            EXPECT_FALSE(idxs.at(idx)) << "Duplicate graphFileIndex";
            idxs.at(idx) = true;
        });

    for (auto b : idxs)
    {
        EXPECT_TRUE(b) << "Unused graphFileIndex";
    }
}

// Verify that the highest-priority tensor is still highest-priority with reduced CSRAM.
TEST_F(LayoutDMATest, high_priority_preserved)
{
    std::string name;
    unsigned size = 0;

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.initialize();
        unit.run();

        forEachGraphfileTensor(unit.model(), [&](mv::Tensor* t, mv::Tensor* p)
            {
                unsigned idx = t->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = p;
                }
            });

        ASSERT_NE(tensor, nullptr);

        name = tensor->getName();
        size = tensor->size();

        EXPECT_GT(size, 0);
    }

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.compilationDescriptor().setPassArg("LayoutDMA", "csramLimit", static_cast<int>(size));
        unit.initialize();
        unit.run();

        forEachGraphfileTensor(unit.model(), [&](mv::Tensor* t, mv::Tensor* p)
            {
                unsigned idx = t->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = p;
                }
            });

        ASSERT_NE(tensor, nullptr);

        EXPECT_EQ(tensor->getName(), name);
    }
}

// Verify that the highest-priority tensor is not the highest priority
// when there's insufficient CSRAM.  (NB this depends on the
// particular network we happen to be using.)
TEST_F(LayoutDMATest, alternative_high_priority)
{
    std::string name;
    unsigned size = 0;

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.initialize();
        unit.run();

        forEachGraphfileTensor(unit.model(), [&](mv::Tensor* t, mv::Tensor* p)
            {
                unsigned idx = t->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = p;
                }
            });

        ASSERT_NE(tensor, nullptr);

        name = tensor->getName();
        size = tensor->size();

        EXPECT_GT(size, 1);
    }

    {
        mv::Tensor* tensor = nullptr;
        mv::CompilationUnit unit{"conv"};
        LoadModel(&unit);
        unit.compilationDescriptor().setPassArg("LayoutDMA", "csramLimit", static_cast<int>(size - 1));
        unit.initialize();
        unit.run();

        forEachGraphfileTensor(unit.model(), [&](mv::Tensor* t, mv::Tensor* p)
            {
                unsigned idx = t->get<unsigned>("graphFileIndex");
                if (!idx)
                {
                    tensor = p;
                }
            });

        ASSERT_NE(tensor, nullptr);

        EXPECT_NE(tensor->getName(), name);
    }
}

}  // namespace
