// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <filesystem>

#include <vector>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <common_test_utils/test_constants.hpp>
#include <openvino/core/any.hpp>
#include <openvino/core/memory_util.hpp>
#include <openvino/core/type/element_iterator.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {
class NPUInferRequestDynamicRemoteTensorsTests_NPU3720 : public OVInferRequestDynamicTests {
protected:
    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = ie->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        ov::InferRequest req;
        req = net.create_infer_request();
        auto tensor = req.get_tensor(function->inputs().back().get_any_name());
        tensor.set_shape(in.get_shape());
        for (size_t i = 0; i < in.get_size(); i++) {
            tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[i] =
                    in.data<ov::element_type_traits<ov::element::f32>::value_type>()[i];
        }
        req.infer();
        OVInferRequestDynamicTests::checkOutput(actual, req.get_output_tensor(0));
    }
};

TEST_P(NPUInferRequestDynamicRemoteTensorsTests_NPU3720, InferDynamicNetworkRemoteTensorLinux) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (!std::filesystem::exists("/dev/dma_heap/system")) {
        OPENVINO_THROW("Cannot open /dev/dma_heap/system file.");
    }

    auto fd_dma_heap = open("/dev/dma_heap/system", O_RDWR);
    if (fd_dma_heap == -1) {
        OPENVINO_THROW("Cannot open /dev/dma_heap/system.");
    }

    static const std::size_t alignment = 4096;
    std::vector<ov::Shape> vector_shapes{inOutShapes[0].first, inOutShapes[0].first};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(1, inOutShapes[1].first[0]), ov::Dimension(1, inOutShapes[1].first[1]),
                         ov::Dimension(1, inOutShapes[1].first[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto context = ie->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto inference_request = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";
    for (auto& shape : vector_shapes) {
        ov::Tensor in_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

        const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(in_tensor.get_shape()));
        size_t size = byte_size + alignment - (byte_size % alignment);
        struct dma_heap_allocation_data heapAlloc = {
                .len = size,  // this length should be alligned to the page size
                .fd = 0,
                .fd_flags = O_RDWR | O_CLOEXEC,
                .heap_flags = 0,
        };
        auto ret = ioctl(fd_dma_heap, DMA_HEAP_IOCTL_ALLOC, &heapAlloc);
        if (ret != 0) {
            OPENVINO_THROW("Cannot initialize DMA heap");
        }
        auto fd_heap = static_cast<int32_t>(heapAlloc.fd);
        auto mmap_ret = mmap(NULL, byte_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd_heap, 0);
        if (mmap_ret == MAP_FAILED) {
            ASSERT_FALSE(true) << "mmap failed.";
        }
        memcpy(mmap_ret, in_tensor.data(), byte_size);

        auto remote_tensor = context.create_tensor(ov::element::f32, in_tensor.get_shape(), fd_heap);

        OV_ASSERT_NO_THROW(req = inference_request.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor(inputName, remote_tensor));
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(in_tensor, req.get_tensor(outputName)));
        close(fd_heap);
    }
    close(fd_dma_heap);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

#endif
#endif
