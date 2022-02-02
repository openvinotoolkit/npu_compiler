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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct ProposalParams final {
    ProposalParams& feat_stride(size_t feat_stride) {
        feat_stride_ = feat_stride;
        return *this;
    }

    ProposalParams& base_size(size_t base_size) {
        base_size_ = base_size;
        return *this;
    }

    ProposalParams& min_size(size_t min_size) {
        min_size_ = min_size;
        return *this;
    }

    ProposalParams& pre_nms_topn(int pre_nms_topn) {
        pre_nms_topn_ = pre_nms_topn;
        return *this;
    }

    ProposalParams& post_nms_topn(int post_nms_topn) {
        post_nms_topn_ = post_nms_topn;
        return *this;
    }

    ProposalParams& nms_thresh(float nms_thresh) {
        nms_thresh_ = nms_thresh;
        return *this;
    }

    ProposalParams& framework(std::string framework) {
        framework_ = std::move(framework);
        return *this;
    }

    ProposalParams& scale(std::vector<float> scale) {
        scale_ = std::move(scale);
        return *this;
    }

    ProposalParams& ratio(std::vector<float> ratio) {
        ratio_ = std::move(ratio);
        return *this;
    }

    ProposalParams& normalize(bool normalize) {
        normalize_ = normalize;
        return *this;
    }

    ProposalParams& for_deformable(bool for_deformable) {
        for_deformable_ = for_deformable;
        return *this;
    }

    ProposalParams& clip_after_nms(bool clip_after_nms) {
        clip_after_nms_ = clip_after_nms;
        return *this;
    }

    ProposalParams& clip_before_nms(bool clip_before_nms) {
        clip_before_nms_ = clip_before_nms;
        return *this;
    }

    ProposalParams& box_scale(float box_size_scale) {
        box_size_scale_ = box_size_scale;
        return *this;
    }

    ProposalParams& box_coord_scale(float box_coordinate_scale) {
        box_coordinate_scale_ = box_coordinate_scale;
        return *this;
    }

    size_t feat_stride_ = 0;
    size_t base_size_   = 0;
    size_t min_size_    = 0;

    int pre_nms_topn_  = -1;
    int post_nms_topn_ = -1;

    float nms_thresh_           = 0.0;
    float box_coordinate_scale_ = 1.0;
    float box_size_scale_       = 1.0;

    bool normalize_       = false;
    bool clip_before_nms_ = true;
    bool clip_after_nms_  = false;
    bool for_deformable_  = false;

    std::string framework_;
    std::vector<float> scale_;
    std::vector<float> ratio_;
};

template <typename Stream>
inline Stream& operator<<(Stream& os, const ProposalParams& p) {
    vpux::printTo(os, "[feat_stride:{0}, base_size:{1}, min_size:{2}, pre_nms_topn:{3},"
                           "post_nms_topn:{4}, nms_thresh:{5}, box_coordinate_scale:{6},"
                           "box_size_scale:{7}, normalize:{8}, clip_before_nms:{9}, clip_after_nms:{10},"
                           " for_deformable:{11}, framework:{12}, scale:{13}, ratio:{14}]",
                           p.feat_stride_, p.base_size_, p.min_size_, p.pre_nms_topn_,
                           p.post_nms_topn_, p.nms_thresh_, p.box_coordinate_scale_,
                           p.box_size_scale_, p.normalize_, p.clip_before_nms_, p.clip_after_nms_,
                           p.for_deformable_, p.framework_, p.scale_, p.ratio_);
    return os;
}

struct ProposalLayerDef final {
    TestNetwork&   net_;
    std::string    name_;
    ProposalParams params_;

    PortInfo cls_score_port_;
    PortInfo bbox_pred_port_;
    PortInfo img_info_port_;

    ProposalLayerDef(TestNetwork& net, std::string name, ProposalParams params)
        : net_(net), name_(std::move(name)), params_(std::move(params)) {
    }

    ProposalLayerDef& scores(const std::string& layer_name, size_t index = 0) {
        cls_score_port_ = PortInfo(layer_name, index);
        return *this;
    }

    ProposalLayerDef& boxDeltas(const std::string& layer_name, size_t index = 0) {
        bbox_pred_port_ = PortInfo(layer_name, index);
        return *this;
    }

    ProposalLayerDef& imgInfo(const std::string& layer_name, size_t index = 0) {
        img_info_port_ = PortInfo(layer_name, index);
        return *this;
    }

    TestNetwork& build();
};
