//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

// clang-format off

#include <ngraph_ops/convolution_ie.hpp>
#include <memory>

class McmConv final : public ngraph::op::ConvolutionIE {
public:
    OPENVINO_OP("McmConv");

    McmConv() = default;

    McmConv(
            const ngraph::Output<ngraph::Node>& data,
            const ngraph::Output<ngraph::Node>& filters,
            const ngraph::Strides& strides,
            const ngraph::CoordinateDiff& pads_begin,
            const ngraph::CoordinateDiff& pads_end,
            const ngraph::Strides& dilations,
            size_t group,
            const ngraph::element::Type& type);

    void setElemType(const ngraph::element::Type& type) {
        _type = type;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

private:
    ngraph::element::Type _type;
};

// clang-format on
