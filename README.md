<div align="center">

<img src="src/vpux_compiler/docs/guides/images/badges-core-ultra-processor-family-5-7-9-center.png" width="400px">

# OpenVINO™ Intel® NPU Compiler

[![ubuntu_22](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/ubuntu_22.yml/badge.svg?branch=develop&event=push)](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/ubuntu_22.yml?query=event%3Apush)
[![ubuntu_24](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/ubuntu_24.yml/badge.svg?branch=develop&event=push)](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/ubuntu_24.yml?query=event%3Apush)
[![Windows (2022)](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/windows_2022.yml/badge.svg?branch=develop&event=push)](https://github.com/openvinotoolkit/npu_compiler/actions/workflows/windows_2022.yml?query=event%3Apush)

[![OpenVINO Docs](https://img.shields.io/badge/OpenVINO-documentation-blue)](https://docs.openvino.ai/)
[![OpenVINO Downloads](https://img.shields.io/badge/OpenVINO-downloads-blue)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
[![OpenVINO Repository](https://img.shields.io/badge/OpenVINO-repository-blue)](https://github.com/openvinotoolkit/openvino)
[![NPU Plugin](https://img.shields.io/badge/OpenVINO-NPU_Plugin-purple)](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu)

[![NPU Windows Driver](https://img.shields.io/badge/NPU-Windows_Drivers-lightblue)](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
[![NPU Linux Driver](https://img.shields.io/badge/NPU-Linux_Drivers-lightblue)](https://github.com/intel/linux-npu-driver/releases)

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/openvinotoolkit/npu_compiler/badge)](https://scorecard.dev/viewer/?uri=github.com/openvinotoolkit/npu_compiler)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10402/badge)](https://www.bestpractices.dev/projects/10402)

</div>

## References

Check out the [main OpenVINO guide](https://github.com/openvinotoolkit/openvino/blob/master/README.md), the [OpenVINO Cheat Sheet](https://docs.openvino.ai/2025/_static/download/OpenVINO_Quick_Start_Guide.pdf) and [Key Features](https://docs.openvino.ai/2025/about-openvino/key-features.html) for a quick reference.

## NPU Compiler guides

Welcome to the OpenVINO™ Intel® NPU Compiler repository. This guide provides a comprehensive introduction to the compiler, including its architecture, setup, and tools. By the end of this guide, you will understand how the compiler works, how to build and use it locally, and how to begin testing, debugging, and contributing to its development.

Contents:

- [Project Structure](src/vpux_compiler/docs/guides/project_structure.md) – Overview of the project and its purpose
- [Building the Project](src/vpux_driver_compiler/docs/) – Step-by-step instructions for building the compiler from source
- [MLIR Primer Tutorial](src/vpux_compiler/docs/guides/primer_mlir.md) and [MLIR Good Practices](src/vpux_compiler/docs/guides/mlir_good_practices.md) – Introduction to MLIR, the foundational framework used by the compiler
- [NPU Primer Tutorial](src/vpux_compiler/docs/guides/primer_vpu.md) – Overview of the Intel® NPU and its integration with the compiler
- [Debugging Guide](src/vpux_compiler/docs/guides/how_to_debug.md) – Tools and techniques for debugging
- [Tools quick guide](src/vpux_driver_compiler/docs/how_to_debug.md) – Tools and techniques for compiling and measuring performance
- [Testing Guide](src/vpux_driver_compiler/docs/how_to_test.md) – Overview of the project’s testing infrastructure and how to use it

## License

OpenVINO™ Toolkit and OpenVINO™ Intel® NPU Compiler are licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
