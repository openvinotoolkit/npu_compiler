#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("DetectionOutputModel");
    mv::OpModel& om = unit.model();
    // Define Params
    auto num_classes = 21;
    auto keep_top_k = 200;
    auto nms_threshold = 0.0;
    auto background_label_id = 0;
    auto top_k = 400;
    auto variance_encoded_in_target = 0;
    auto code_type = "CENTER_SIZE";
    auto share_location = 1;
    auto confidence_threshold = 0.6;
    auto clip_before_nms = 0;
    auto clip_after_nms = 0;
    auto decrease_label_id = 0;
    auto normalized = 1;
    auto input_height = -1;
    auto input_width = -1;
    auto objectness_score = -1;
    std::vector<uint16_t> bboxPredData(34928);
    std::vector<uint16_t> proposalsData(34928*2);

    //Load BBoxPreds1 tensor from file
    std::string BBoxPreds1_filename(mv::utils::projectRootPath() + "/tests/layer/detection_output/detection_output.in2");
    std::ifstream c_file;
    c_file.open(BBoxPreds1_filename, std::fstream::in | std::fstream::binary);
    c_file.read((char*)(bboxPredData.data()), 34928 * sizeof(uint16_t));
    std::vector<int64_t> bboxPredData_converted(34928);
    for(unsigned i = 0; i < bboxPredData.size(); ++i)
        bboxPredData_converted[i] = bboxPredData[i];
    //Load Proposals1 tensor from file
    std::string Proposals1_filename(mv::utils::projectRootPath() + "/tests/layer/detection_output/detection_output.in3");
    std::ifstream p_file;
    p_file.open(Proposals1_filename, std::fstream::in | std::fstream::binary);
    p_file.read((char*)(proposalsData.data()), 34928 * 2 * sizeof(uint16_t));
    std::vector<int64_t> proposalsData_converted(34928*2);
    for(unsigned i = 0; i < proposalsData.size(); ++i)
        proposalsData_converted[i] = proposalsData[i];
    // Define tensors
    auto bbox_pred = om.constantInt("bbox_pred0", bboxPredData_converted, {34928,1,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    auto cls_pred = om.input("cls_pred0", {183372,1,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    auto proposals = om.constantInt("proposals", proposalsData_converted, {34928,2,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    bbox_pred->setQuantParams({{0},{1.0},{},{}});
    cls_pred->setQuantParams({{0},{1.0},{},{}});
    proposals->setQuantParams({{0},{1.0},{},{}});
    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(bbox_pred);
    inputs.push_back(cls_pred);
    inputs.push_back(proposals);
    // Build Model
    auto detection0 = om.detectionOutput("detection_output", inputs, num_classes, keep_top_k, nms_threshold, background_label_id, top_k, variance_encoded_in_target,
                                                                         code_type, share_location, confidence_threshold, clip_before_nms, clip_after_nms,
                                                                         decrease_label_id, normalized, input_height, input_width, objectness_score);
    detection0->setQuantParams({{128},{0.007843137718737125},{-1.0},{1.0}});
    om.output("", detection0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
