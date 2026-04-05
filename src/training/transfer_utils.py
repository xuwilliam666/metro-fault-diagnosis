from pathlib import Path

import torch


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint does not contain a valid state_dict")

    cleaned = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _inflate_input_weight(pretrained_weight: torch.Tensor, target_shape):
    if pretrained_weight.ndim != len(target_shape):
        return None

    if pretrained_weight.shape == target_shape:
        return pretrained_weight

    if pretrained_weight.ndim == 3:
        out_channels, in_channels, kernel_size = pretrained_weight.shape
        tgt_out, tgt_in, tgt_kernel = target_shape
        if out_channels == tgt_out and kernel_size == tgt_kernel and in_channels == 1:
            return pretrained_weight.repeat(1, tgt_in, 1) / float(tgt_in)

    if pretrained_weight.ndim == 2:
        out_features, in_features = pretrained_weight.shape
        tgt_out, tgt_in = target_shape
        if out_features == tgt_out and in_features == 1:
            return pretrained_weight.repeat(1, tgt_in) / float(tgt_in)

    return None


def load_cwru_pretrained_weights(model, checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    pretrained_state = _extract_state_dict(checkpoint)
    model_state = model.state_dict()

    updated_state = {}
    for key, value in pretrained_state.items():
        if key in ("fc.weight", "fc.bias"):
            continue

        if key not in model_state:
            continue

        target_value = model_state[key]
        candidate = value

        if key in ("conv1.weight", "lstm_in.weight"):
            candidate = _inflate_input_weight(value, target_value.shape)
            if candidate is None:
                continue

        if candidate.shape != target_value.shape:
            continue

        updated_state[key] = candidate

    model.load_state_dict(updated_state, strict=False)
    return model


def freeze_feature_extractor(model, freeze=True):
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze
    return model


def freeze_backbone_layers(model, layer_prefixes):
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            param.requires_grad = False
    return model


def freeze_deeper_feature_layers(model):
    deeper_prefixes = (
        "conv2",
        "bn2",
        "conv3",
        "bn3",
        "lstm",
    )
    return freeze_backbone_layers(model, deeper_prefixes)
