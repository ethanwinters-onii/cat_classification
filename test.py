import torch
import torch.nn as nn
from cait_model import cait_models
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from functools import partial


@register_model
def cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model

if __name__ == "__main__":
    model = cait_XXS24()
    print(model)