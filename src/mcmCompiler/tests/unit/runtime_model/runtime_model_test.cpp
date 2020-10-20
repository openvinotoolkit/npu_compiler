#include "gtest/gtest.h"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "tools/graph_comparator/include/graph_comparator/graph_comparator.hpp"

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

TEST(runtime_model, test_soh_dma_addresses)
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("conv1#34", {416,416,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool("pool1/max_pool#35", input0, {2, 2}, {2, 2}, {0, 0, 0, 0}, true);
    pool0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*16*32);
    auto weights0 = om.constantInt("conv2#4_weights#5", weightsData0,{3,3,16,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights0->setQuantParams({{119},{0.002742463955655694},{-0.32530343532562256},{0.374024897813797}});
    auto conv0 = om.conv("conv2#36", pool0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt("conv2#4_bias#6", biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights0->setQuantParams({{0},{1.075476120604435e-05},{-inf},{inf}});
    auto bias_c0 = om.bias("", conv0, biasWeights0);
    bias_c0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool1 = om.maxPool("pool2/max_pool#37", bias_c0, {2, 2}, {2, 2}, {0, 0, 0, 0}, true);
    pool1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    om.output("", pool1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    mv::DataModel dm(om);
    mv::tools::GraphComparator gc;
    std::string outputBlobFile = "./output/mcm.blob"; // generated blob file

    //XXX: buffer needs to be deleted after the test, or else there will be a leak
    char* buffer = 0;
    const MVCNN::GraphFileT& blob = gc.loadGraphFile(outputBlobFile, buffer);

    std::vector<mv::Data::OpListIterator> dpuTasks;
    for (auto op = om.opBegin(); op != om.opEnd(); ++op)
    {
        if (op->getOpType() == "DPUTask" && op->get<std::string>("splitStrategy") == "SplitOverH")
        {
            dpuTasks.push_back(op);
        }
    }

    for (auto dpuTask: dpuTasks)
    {
        auto inputTensor = dpuTask->getInputTensor(0);
        auto dmaTask = om.getSourceOp(inputTensor);
        auto dmaTaskSrcTensor = dmaTask->getInputTensor(0);
        auto dmaTaskDstTensor = inputTensor;

        for (std::size_t i = 0; i < blob.task_lists.size(); ++i)
        {
            const std::unique_ptr<MVCNN::TaskListT>& task_list = blob.task_lists[i];
            for (std::size_t j = 0; j < task_list->content.size(); j++)
            {
                const std::unique_ptr<MVCNN::TaskT>& task = task_list->content[j];
                if (task->name == dmaTask->getName() && task->task.type == MVCNN::SpecificTask::SpecificTask_NNDMATask)
                {
                    const MVCNN::NNDMATaskT& blobDmaTask = *task->task.AsNNDMATask();
                    for (int k = 0; k < 2; k++)
                    {
                        std::string blobTaskName;
                        mv::Data::TensorIterator expectedTensor;
                        MVCNN::IndirectDataReferenceT *blobTensorIndirectRef;
                        if (k == 0)
                        {
                            blobTaskName = blobDmaTask.src->name;
                            expectedTensor = dmaTaskSrcTensor;
                            blobTensorIndirectRef = blobDmaTask.src->data.get();
                        }
                        else
                        {
                            blobTaskName = blobDmaTask.dst->name;
                            expectedTensor = inputTensor;
                            blobTensorIndirectRef = blobDmaTask.dst->data.get();
                        }

                        std::string subTensorId = blobTaskName.substr(blobTaskName.find("sub") + 3);
                        std::size_t clusterId = std::stoi(subTensorId);

                        auto subtensor = expectedTensor->getSubTensor(clusterId);
                        auto tensorAllocators = expectedTensor->get<std::set<std::string>>("allocators");
                        auto tensorAllocatorName = tensorAllocators.begin();
                        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);

                        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, expectedTensor);
                        std::vector<uint32_t> dimensions = subtensor.getShape();

                        if (*tensorAllocatorName == "GraphFile")
                        {
                            if(!dmaTaskSrcTensor->isSparse())
                            {

                                auto offset = subtensor.get<std::vector<std::size_t>>("offset");
                                auto index = expectedTensor->getOrder().subToInd(expectedTensor->getShape(), offset);
                                auto byte_index = index * expectedTensor->getDType().getSizeInBits() / 8;

                                ASSERT_EQ(blobTensorIndirectRef->data_index, byte_index);
                            }
                            else
                            {
                                ASSERT_EQ(blobTensorIndirectRef->data_index, 0);
                            }
                        }
                        else if(*tensorAllocatorName == "ProgrammableInput" || *tensorAllocatorName == "ProgrammableOutput" ||
                                *tensorAllocatorName == "VPU_DDR_BSS" || *tensorAllocatorName == "VPU_DDR_Heap")
                        {
                            auto offset = subtensor.get<std::vector<std::size_t>>("offset");
                            auto index = expectedTensor->getOrder().subToInd(expectedTensor->getShape(), offset);
                            auto byte_index = index * expectedTensor->getDType().getSizeInBits() / 8;

                            auto starting_address = 0;
                            if(expectedTensor->hasAttr("address"))
                                starting_address = expectedTensor->get<std::size_t>("address");
                            else
                            {
                                auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
                                starting_address = (*masterBuffer)->getOffset();
                            }

                            ASSERT_EQ(blobTensorIndirectRef->data_index, starting_address + byte_index);

                        }
                        else
                        {
                            if(expectedTensor->hasAttr("address"))
                                ASSERT_EQ(blobTensorIndirectRef->data_index, subtensor.getAddress());
                            else
                                ASSERT_EQ(blobTensorIndirectRef->data_index, tensorBufferIt->getOffset());
                        }
                    }

                }
            }
        }
    }

    if (buffer)
        delete [] buffer;

}


TEST(runtime_model, test_hkswitch_address_assignment)
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input("model/re_lu_5/Relu#85", {52,52,256,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{0.07285668700933456},{0.0},{18.578454971313477}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*256*128);
    auto weights0 = om.constantInt("model/re_lu_6/Relu#21_weights#22", weightsData0,{1,1,256,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));    auto conv0 = om.conv("model/re_lu_6/Relu#86", input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    weights0->setQuantParams({{139},{0.01167607493698597},{-1.6125890016555786},{1.353134036064148}});
    conv0->setQuantParams({{0},{0.09908168017864227},{0.0},{25.26582908630371}});
    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights0 = om.constantInt("model/re_lu_6/Relu#21_bias#23", biasWeightsData0,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));     auto bias_c0 = om.bias("", conv0, biasWeights0);
    biasWeights0->setQuantParams({{0},{0.0008506801095791161},{-inf},{inf}});
    bias_c0->setQuantParams({{0},{0.09908168017864227},{0.0},{25.26582908630371}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*128*256);
    auto weights1 = om.constantInt("model/re_lu_7/Relu#24_weights#25", weightsData1,{3,3,128,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));    auto conv1 = om.conv("model/re_lu_7/Relu#87", bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1);
    weights1->setQuantParams({{114},{0.0023650287184864283},{-0.2681608498096466},{0.33255642652511597}});
    conv1->setQuantParams({{0},{0.07538475096225739},{0.0},{19.223112106323242}});
    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights1 = om.constantInt("model/re_lu_7/Relu#24_bias#26", biasWeightsData1,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));     auto bias_c1 = om.bias("", conv1, biasWeights1);
    biasWeights1->setQuantParams({{0},{0.0002343310188734904},{-inf},{inf}});
    bias_c1->setQuantParams({{0},{0.0002343310188734904},{-inf},{inf}});


    auto pool0 = om.maxPool("model/max_pooling2d_3/MaxPool#88", bias_c1, {2, 2}, {2, 2}, {0, 0, 0, 0}, true);
    pool0->setQuantParams({{0},{0.07538475096225739},{0.0},{19.223112106323242}});
    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*256*512);
    auto weights2 = om.constantInt("model/re_lu_8/Relu#28_weights#29", weightsData2,{3,3,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));    auto conv2 = om.conv("model/re_lu_8/Relu#89", pool0, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1);
    weights2->setQuantParams({{73},{0.0030357716605067253},{-0.21838819980621338},{0.5526977777481079}});
    conv2->setQuantParams({{0},{0.0771605372428894},{0.0},{19.67593765258789}});
    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights2 = om.constantInt("model/re_lu_8/Relu#28_bias#30", biasWeightsData2,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));    auto bias_c2 = om.bias("", conv2, biasWeights2);
    biasWeights2->setQuantParams({{0},{0.00022885088401380926},{-inf},{inf}});
    bias_c2->setQuantParams({{0},{0.0771605372428894},{0.0},{19.67593765258789}});

    om.output("", bias_c2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();


    mv::DataModel dm(om);
    mv::tools::GraphComparator gc;
    std::string outputBlobFile = "./output/mcm.blob"; // generated blob file

    //XXX: buffer needs to be deleted after the test, or else there will be a leak
    char* buffer = 0;
    const MVCNN::GraphFileT& blob = gc.loadGraphFile(outputBlobFile, buffer);

    std::vector<mv::Data::OpListIterator> dpuTasks;
    for (auto op = om.opBegin(); op != om.opEnd(); ++op)
    {
        if (op->getOpType() == "DPUTask" && op->get<std::string>("splitStrategy") == "HKSwitch")
        {
            std::cout << "found hkswitch op" << op->getName() << std::endl;
            dpuTasks.push_back(op);
        }
    }

    for (auto dpuTask: dpuTasks)
    {
        // auto inputTensor = dpuTask->getInputTensor(0);
        // auto dmaTask = om.getSourceOp(inputTensor);
        // auto dmaTaskSrcTensor = dmaTask->getInputTensor(0);
        // auto dmaTaskDstTensor = inputTensor;

        for (std::size_t i = 0; i < blob.task_lists.size(); ++i)
        {
            const std::unique_ptr<MVCNN::TaskListT>& task_list = blob.task_lists[i];
            for (std::size_t j = 0; j < task_list->content.size(); j++)
            {
                const std::unique_ptr<MVCNN::TaskT>& task = task_list->content[j];
                if (task->name == dpuTask->getName() && task->task.type == MVCNN::SpecificTask::SpecificTask_NCE2Task)
                {
                    MVCNN::NCE2TaskT& blobDpuTask = *task->task.AsNCE2Task();
                    MVCNN::NCEInvariantFieldsT *blobNCEInvariantFields = blobDpuTask.invariant.get();
                    std::string blobOutputDataName = blobNCEInvariantFields->output_data->name;
                    std::string subTensorId = blobOutputDataName.substr(blobOutputDataName.find("sub") + 3);
                    std::size_t clusterId = std::stoi(subTensorId);

                    auto expectedTensor = dpuTask->getOutputTensor(0);
                    auto expectedSubtensor = expectedTensor->getSubTensor(clusterId);
                    auto tensorAllocators = expectedTensor->get<std::set<std::string>>("allocators");
                    auto tensorAllocatorName = tensorAllocators.begin();
                    ASSERT_EQ(*tensorAllocatorName,"VPU_CMX_NN");

                    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
                    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, expectedTensor);

                    uint64_t expectedAddress = 0;
                    if(expectedTensor->hasAttr("address"))
                        expectedAddress = expectedSubtensor.getAddress();
                    else
                        expectedAddress = tensorBufferIt->getOffset();

                    auto offset = expectedSubtensor.get<std::vector<std::size_t>>("offset");
                    auto index = expectedTensor->getOrder().subToInd(expectedTensor->getShape(), offset);
                    auto byte_index = index * expectedTensor->getDType().getSizeInBits() / 8;

                    expectedAddress += byte_index;
                    ASSERT_EQ(blobNCEInvariantFields->output_data->data->data_index, expectedAddress);
                }
            }
        }
    }

    if (buffer)
        delete [] buffer;

}
