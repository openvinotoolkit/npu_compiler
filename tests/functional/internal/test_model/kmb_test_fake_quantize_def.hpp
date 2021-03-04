//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct FakeQuantizeLayerDef final {
    TestNetwork& testNet;

    std::string name;

    PortInfo inputPort;
    PortInfo inputLowPort;
    PortInfo inputHighPort;
    PortInfo outputLowPort;
    PortInfo outputHighPort;
    size_t levels;

    FakeQuantizeLayerDef(TestNetwork& testNet, std::string name, size_t levels)
        : testNet(testNet), name(std::move(name)), levels(levels) {
    }

    FakeQuantizeLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& input(const Blob::Ptr& blob) {
        const auto inputLayerName = name + "_input";
        testNet.addConst(inputLayerName, blob);
        return input(inputLayerName);
    }

    FakeQuantizeLayerDef& inputLow(const std::string& layerName, size_t index = 0) {
        inputLowPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& inputLow(const Blob::Ptr& blob) {
        const auto inputLowLayerName = name + "_inputLow";
        testNet.addConst(inputLowLayerName, blob);
        return inputLow(inputLowLayerName);
    }
    FakeQuantizeLayerDef& inputLow(float val, const Precision& precision) {
        const auto inputLowBlob = vpux::makeScalarBlob(val, precision);
        return inputLow(inputLowBlob);
    }

    FakeQuantizeLayerDef& inputHigh(const std::string& layerName, size_t index = 0) {
        inputHighPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& inputHigh(const Blob::Ptr& blob) {
        const auto inputHighLayerName = name + "_inputHigh";
        testNet.addConst(inputHighLayerName, blob);
        return inputHigh(inputHighLayerName);
    }
    FakeQuantizeLayerDef& inputHigh(float val, const Precision& precision) {
        const auto inputHighBlob = vpux::makeScalarBlob(val, precision);
        return inputHigh(inputHighBlob);
    }

    FakeQuantizeLayerDef& outputLow(const std::string& layerName, size_t index = 0) {
        outputLowPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& outputLow(const Blob::Ptr& blob) {
        const auto outputLowLayerName = name + "_outputLow";
        testNet.addConst(outputLowLayerName, blob);
        return outputLow(outputLowLayerName);
    }
    FakeQuantizeLayerDef& outputLow(float val, const Precision& precision) {
        const auto outputLowBlob = vpux::makeScalarBlob(val, precision);
        return outputLow(outputLowBlob);
    }

    FakeQuantizeLayerDef& outputHigh(const std::string& layerName, size_t index = 0) {
        outputHighPort = PortInfo(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& outputHigh(const Blob::Ptr& blob) {
        const auto outputHighLayerName = name + "_outputHigh";
        testNet.addConst(outputHighLayerName, blob);
        return outputHigh(outputHighLayerName);
    }
    FakeQuantizeLayerDef& outputHigh(float val, const Precision& precision) {
        const auto outputHighBlob = vpux::makeScalarBlob(val, precision);
        return outputHigh(outputHighBlob);
    }

    FakeQuantizeLayerDef& low(const std::string& layerName, size_t index = 0) {
        inputLow(layerName, index);
        outputLow(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& low(const Blob::Ptr& blob) {
        const auto lowLayerName = name + "_low";
        testNet.addConst(lowLayerName, blob);
        inputLow(lowLayerName);
        outputLow(lowLayerName);
        return *this;
    }
    FakeQuantizeLayerDef& low(float val, const Precision& precision) {
        const auto lowBlob = vpux::makeScalarBlob(val, precision);
        inputLow(lowBlob);
        outputLow(lowBlob);
        return *this;
    }

    FakeQuantizeLayerDef& high(const std::string& layerName, size_t index = 0) {
        inputHigh(layerName, index);
        outputHigh(layerName, index);
        return *this;
    }
    FakeQuantizeLayerDef& high(const Blob::Ptr& blob) {
        const auto highLayerName = name + "_high";
        testNet.addConst(highLayerName, blob);
        inputHigh(highLayerName);
        outputHigh(highLayerName);
        return *this;
    }
    FakeQuantizeLayerDef& high(float val, const Precision& precision) {
        const auto highBlob = vpux::makeScalarBlob(val, precision);
        inputHigh(highBlob);
        outputHigh(highBlob);
        return *this;
    }

    TestNetwork& build();
};
