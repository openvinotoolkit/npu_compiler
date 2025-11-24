"""
Converts a PyTorch model to ONNX format
"""

import requests
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    CLIPProcessor,
    CLIPModel,
    SamProcessor,
    SamModel,
    ResNetForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)


class CLIPWrapper(torch.nn.Module):
    """
    CLIPModel returns CLIPOutput objects (dict-like).
    ONNX export expects a tuple of tensors, but get tuple of dicts.
    This class wraps CLIPModel and returns unfolded outputs as a tuple of 4 tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        """Unfolds CLIPOutput into tensors"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        return (
            outputs.logits_per_image,
            outputs.logits_per_text,
            outputs.text_embeds,
            outputs.image_embeds,
        )


class PromptEncoderWrapper(torch.nn.Module):
    """
    SamPromptEncoder takes extra input_boxes optional input
    that shouldn't be counted in onnx model inputs.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_points, input_labels, input_masks):
        """Adds extra input_boxes input"""
        sparse_embeddings, dense_embeddings = self.encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_masks=input_masks,
            input_boxes=None,
        )
        return sparse_embeddings, dense_embeddings


class MaskDecoderWrapper(torch.nn.Module):
    """
    SamMaskDecoder takes extra image_positional_embeddings input
    that shouldn't be counted in onnx model inputs.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        image_embeddings,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        """Adds extra image_positional_embeddings input"""
        low_res_masks, iou_predictions = self.decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=torch.randn_like(image_embeddings),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


def convert_pytorch_to_onnx(model: dict, torch_model_path: str):
    """Converts a PyTorch model to ONNX format"""
    pytorch_model_dir = torch_model_path.parent
    onnx_model_path = pytorch_model_dir / f"{model.get('name')}.onnx"
    model_type = model.get("model_type", "")
    convert_config = model.get("Convert", {})

    input_names = convert_config.get("input_names", [])
    output_names = convert_config.get("output_names", [])

    dynamic_axes = {}
    for name, shape in convert_config.get("dynamic_axes", {}).items():
        dynamic_axes[name] = {int(axis): type for axis, type in shape.items()}

    if model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir, use_fast=True)
        pytorch_model = AutoModel.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()

        text1 = convert_config.get("text1", "")
        text2 = convert_config.get("text2", "")
        inputs = tokenizer(text1, text2, return_tensors="pt")
    elif model_type == "vision_transformer":
        processor = ViTImageProcessor.from_pretrained(pytorch_model_dir, use_fast=True)
        pytorch_model = ViTForImageClassification.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        inputs = processor(image, return_tensors="pt")
    elif model_type == "clip":
        pytorch_model = CLIPModel.from_pretrained(pytorch_model_dir)
        pytorch_model = CLIPWrapper(pytorch_model).eval()
        processor = CLIPProcessor.from_pretrained(pytorch_model_dir)

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        text1 = convert_config.get("text1", "")
        inputs = processor(
            text=text1, images=[image], return_tensors="pt", padding=True
        )
    elif model_type == "sam_image_encoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        processor = SamProcessor.from_pretrained(pytorch_model_dir)
        pytorch_model = sam_pytorch_model.vision_encoder.eval()

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        ).convert("RGB")
        inputs = processor(image, return_tensors="pt")
    elif model_type == "sam_prompt_encoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        processor = SamProcessor.from_pretrained(pytorch_model_dir)
        pytorch_model = PromptEncoderWrapper(sam_pytorch_model.prompt_encoder.eval())

        point_coords_values = [float(x) for x in convert_config.get("point_coords", [])]
        point_labels_values = [int(x) for x in convert_config.get("point_labels", [])]
        point_coords = torch.tensor([[[point_coords_values]]], dtype=torch.float)
        point_labels = torch.tensor([[point_labels_values]], dtype=torch.int)
        mask_input = torch.zeros(
            tuple(convert_config.get("mask_input", [])), dtype=torch.float
        )
        inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
        }
    elif model_type == "sam_mask_decoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        pytorch_model = MaskDecoderWrapper(sam_pytorch_model.mask_decoder.eval())

        image_embeddings = torch.rand(convert_config.get("image_embeddings_shape", []))
        sparse_embeddings = torch.rand(
            convert_config.get("sparse_embeddings_shape", [])
        )
        dense_embeddings = torch.rand(convert_config.get("dense_embeddings_shape", []))
        multimask_output = torch.tensor([0], dtype=torch.bool)
        inputs = {
            "image_embeddings": image_embeddings,
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
            "multimask_output": multimask_output,
        }
    elif model_type == "resnet":
        pytorch_model = ResNetForImageClassification.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()
        processor = AutoImageProcessor.from_pretrained(pytorch_model_dir, use_fast=True)

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        inputs = processor(image, return_tensors="pt")

    torch.onnx.export(
        pytorch_model,
        args=tuple(inputs.get(input_name) for input_name in input_names),
        f=str(onnx_model_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )

    return onnx_model_path
