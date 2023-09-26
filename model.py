import torch
import torch.nn as nn
from torchvision import models
import config as cfg
from cait_model import cait_models
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from functools import partial
from torchsummary import summary

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

################################################################################
#################################   CaiT XXS24   ###############################
################################################################################

@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

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


################################################################################
#################################   CaiT XXS36   ###############################
################################################################################



@register_model
def cait_XXS36_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_XXS36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 


################################################################################
#################################   CaiT XS24   ###############################
################################################################################


@register_model
def cait_XS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model


################################################################################
#################################   CaiT S24   #################################
################################################################################ 


@register_model
def cait_S24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
            map_location="cpu", check_hash=True
        )

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 



@register_model
def cait_S24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 


################################################################################
#################################   CaiT S36   #################################
################################################################################


@register_model
def cait_S36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


################################################################################
#################################   CaiT M36   #################################
################################################################################


@register_model
def cait_M36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


################################################################################
#################################   CaiT M48   #################################
################################################################################


@register_model
def cait_M48(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 448 , patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model        

################################################################################
#################################   init model   #################################
################################################################################ 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    model_ft = None
    input_size = 0

    #--------------------------- CONVOLUTION NEURAL NETWORK --------------------------------------------#  

    if model_name == 'resnet':
        """
            Resnet18
            (fc): Linear(in_features=512, out_features=1000, bias=True)
            >> model.fc = nn.Linear(512, num_classes)
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == 'alexnet':
        """
        Alexnet
        (classifier): Sequential(
            ...
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
        >> model.classifier[6] = nn.Linear(4096,num_classes)
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vgg':
        """
        VGG11
        (classifier): Sequential(
            ...
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
        >> model.classifier[6] = nn.Linear(4096,num_classes)
        """

        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_channels
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'squeezenet':
        """
        Squeezenet
        (classifier): Sequential(
            (0): Dropout(p=0.5)
            (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
            (2): ReLU(inplace)
            (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
        )
        >> model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_channels
        model_ft.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1,1), stride=(1,1))
        input_size = 224

    elif model_name == "densenet":
        """ 
        Densenet
        (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        >> model.classifier = nn.Linear(1024, num_classes)
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ 
        Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        (AuxLogits): InceptionAux(
            ...
            (fc): Linear(in_features=768, out_features=1000, bias=True)
        )
        ...
        (fc): Linear(in_features=2048, out_features=1000, bias=True)
        >> model.AuxLogits.fc = nn.Linear(768, num_classes)
        >> model.fc = nn.Linear(2048, num_classes)
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "mobilenetv2":
        '''
            (classifier): Sequential(
                (0): Dropout(p=0.2, inplace=False)
                (1): Linear(in_features=1280, out_features=1000, bias=True)
            )
        '''
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        # model_ft.classifier.add_module('softmax', nn.Softmax(dim=1))
        input_size = 224
    
    elif model_name == "mobilenet_v3_small":
        '''
          (classifier): Sequential(
                (0): Linear(in_features=576, out_features=1024, bias=True)
                (1): Hardswish()
                (2): Dropout(p=0.2, inplace=True)
                (3): Linear(in_features=1024, out_features=1000, bias=True)
            )
        '''
        model_ft = models.mobilenet_v3_small(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        input_size = 224

    elif model_name == "mobilenet_v3_large":
        '''
          (classifier): Sequential(
                (0): Linear(in_features=960, out_features=1280, bias=True)
                (1): Hardswish()
                (2): Dropout(p=0.2, inplace=True)
                (3): Linear(in_features=1280, out_features=1000, bias=True)
            )
        '''
        model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        input_size = 224
    
    elif model_name == "efficentnetb7":
        '''
            (classifier): Sequential(
                (0): Dropout(p=0.2, inplace=False)
                (1): Linear(in_features=1280, out_features=1000, bias=True)
            )
        '''
        model_ft = models.efficientnet_b7(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        # model_ft.classifier.add_module('softmax', nn.Softmax(dim=1))
        input_size = 224
    
    #--------------------------- TRANSFORMER --------------------------------------------#  


    elif model_name == "caiT_XXS24_224":
        model_ft = cait_XXS24_224(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 224
    
    elif model_name == "caiT_XXS24":
        model_ft = cait_XXS24(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384

    elif model_name == "caiT_XXS36_224":
        model_ft = cait_XXS36_224(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 224
    
    elif model_name == "caiT_XXS36":
        model_ft = cait_XXS36(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384

    elif model_name == "caiT_XS24":
        model_ft = cait_XS24(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384
    
    elif model_name == "caiT_S24_224":
        model_ft = cait_S24_224(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 224

    elif model_name == "caiT_S24":
        model_ft = cait_S24(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384

    elif model_name == "caiT_S36":
        model_ft = cait_S36(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384
    
    elif model_name == "caiT_M36":
        model_ft = cait_M36(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384
    
    elif model_name == "caiT_M48":
        model_ft = cait_M48(pretrained=use_pretrained)
        num_ftr = model_ft.head.in_features
        model_ft.head = nn.Linear(in_features=num_ftr, out_features=num_classes, bias=True)
        input_size = 384

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

if __name__ == "__main__":
    # Initialize the model for this run
    model_ft, input_size = initialize_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.FEATURE_EXTRACT, use_pretrained=False)

    # Print the model we just instantiated
    # print(model_ft)
    # print(model_ft)

    summary(model_ft.to('cuda'), (3,224,224))




