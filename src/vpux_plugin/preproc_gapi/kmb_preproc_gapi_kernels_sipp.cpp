//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#if defined(__arm__) || defined(__aarch64__)
#include <sippDefs.h>
#include <sippSWConfig.h>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi_sipp/gsippkernel.hpp>

#include "kmb_preproc_gapi_kernels.hpp"

namespace InferenceEngine {
namespace gapi {
namespace preproc {

// clang-format off
// we can't let clang-format tool work with this code. It would ruin everything

GAPI_SIPP_KERNEL(GSippNV12toRGBp, GNV12toRGBp) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::GMatDesc) {
        return {SVU_SYM(svucvtColorNV12toRGB), 0, 1, 1, SIPP_RESIZE};
    }

    static void Configure(cv::GMatDesc, cv::GMatDesc, const cv::GSippConfigUserContext&)
    {}
};

GAPI_SIPP_KERNEL(GSippNV12toBGRp, GNV12toBGRp) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::GMatDesc) {
        return {SVU_SYM(svucvtColorNV12toBGR), 0, 1, 1, SIPP_RESIZE};
    }

    static void Configure(cv::GMatDesc, cv::GMatDesc, const cv::GSippConfigUserContext&)
    {}
};

GAPI_SIPP_KERNEL(GSippResizeP, GResizeP) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::gapi::own::Size, int) {
        int paramSize = sizeof(ScaleBilinearPlanarParams);
        return {SVU_SYM(svuScaleBilinearPlanar), paramSize, 2, 2, SIPP_RESIZE_OPEN_CV, true};
    }

    static void Configure(cv::GMatDesc, cv::gapi::own::Size, int, const cv::GSippConfigUserContext& ctx) {
        ScaleBilinearPlanarParams sclParams;
        sclParams.nChan = 3;
        sclParams.firstSlice = ctx.firstSlice;

        // FIXME?
        // hide in GSippFilter (return param structure)?
        sippSendFilterConfig(ctx.filter, &sclParams, sizeof(ScaleBilinearPlanarParams));
    }
};

// FIXME:
// Rewrite using sipp_kernel_simple
GAPI_SIPP_KERNEL(GSippMerge3p, GMerge3p) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc) {
        return {SVU_SYM(svuMerge3p), 0, 1, 1, 0, false};
    }

    static void Configure(cv::GMatDesc, const cv::GSippConfigUserContext&) {}
};

// FIXME: Kernels below don't have vpu version yet
GAPI_SIPP_KERNEL(GSippScalePlanes, GScalePlanes) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&, const cv::gapi::own::Size&, int) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::gapi::own::Size&, int, const cv::GSippConfigUserContext&) {}
};

GAPI_SIPP_KERNEL(GSippMerge2, GMerge2) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::GSippConfigUserContext&) {}
};

GAPI_SIPP_KERNEL(GSippMerge4, GMerge4) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::GSippConfigUserContext&) {}
};

GAPI_SIPP_KERNEL(GSippDrop4, GDrop4) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::GSippConfigUserContext&) {}
};

GAPI_SIPP_KERNEL(GSippSwapChan, GSwapChan) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::GSippConfigUserContext&) {}
};

GAPI_SIPP_KERNEL(GSippInterleaved2planar, GInterleaved2planar) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(const cv::GMatDesc&) {
        return {};
    }

    static void Configure(const cv::GMatDesc&, const cv::GSippConfigUserContext&) {}
};

namespace sipp {
    cv::gapi::GKernelPackage kernels() {
        static auto pkg = cv::gapi::kernels
            < GSippNV12toBGRp
            , GSippNV12toRGBp
            , GSippMerge3p
            , GSippResizeP
            // FIXME: Kernels below don't have vpu version yet
            , GSippScalePlanes
            , GSippMerge2
            , GSippMerge4
            , GSippDrop4
            , GSippSwapChan
            , GSippInterleaved2planar
            >();
        return pkg;
    }
// clang-format on
}  // namespace sipp
}  // namespace preproc
}  // namespace gapi
}  // namespace InferenceEngine
#endif
