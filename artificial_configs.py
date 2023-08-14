import numpy as np

import torch.nn as nn

class DefaultArgs:
    def __init__(self):
        self.img_set = 'majaj-hong'
        self.num_imgs = 3000
        self.source_model = 'alexnet'
        self.target_model = 'alexnet'
        self.source_pckg = 'pytorch'
        self.target_pckg = 'pytorch'
        self.source_pretrain = False
        self.target_pretrain = True
        self.source_w_path = None
        self.target_w_path = None
        self.source_target_same = False
        self.source_layers = ['classifier.2']
        self.target_layers = ['classifier.2']
        self.source_types = None
        self.target_types = None
        self.srp_bool = True
        self.srp_dim = 1000
        self.num_target_units = 3000 
        self.target_ratio = None
        self.metric = 'ridge'
        self.num_seeds = 1
        self.params = None
        self.outer_cv_splits = 5
        self.inner_cv_splits = 4
        self.gpu_index = 1
        self.savefolder = None
        self.savefile = 'default'
        self.suffix = 'default'

########################################################################################
# target network = AlexNet
########################################################################################
class UnAlexNetAlexNet(DefaultArgs):
    def __init__(self):
        super(UnAlexNetAlexNet, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 'alexnet'
        self.source_pretrain = False
        self.target_pretrain = True
        self.target_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/alexnet/1/checkpoint.pth.tar' 
        self.source_layers = None
        self.target_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unalex-tralex'
        self.num_seeds = 1    
        self.params = [0.001,0.003,0.01,0.03,0.1,0.3]

class TrAlexNetAlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(TrAlexNetAlexNet, self).__init__()
        self.source_pretrain = True
        self.source_target_same = True
        self.savefile = 'tralex-tralex'

class TwoSeedsAlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(TwoSeedsAlexNet, self).__init__()
        self.source_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/alexnet/2/checkpoint.pth.tar' 
        self.savefile = 'tralex1-tralex2'

class UnVgg11AlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(UnVgg11AlexNet, self).__init__()
        self.source_model = 'vgg11'
        self.source_layers = None
        self.source_types = (nn.ReLU)
        self.savefile = 'unvgg11-tralex'

class TrVgg11AlexNet(UnVgg11AlexNet):
    def __init__(self):
        super(TrVgg11AlexNet, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvgg11-tralex'

class UnResnet18AlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(UnResnet18AlexNet, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unres18-tralex'
        #self.num_seeds = 3

class TrResnet18AlexNet(UnResnet18AlexNet):
    def __init__(self):
        super(TrResnet18AlexNet, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trres18-tralex'

class UnViTB32AlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(UnViTB32AlexNet, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'unvitb32-tralex'

class TrViTB32AlexNet(UnViTB32AlexNet):
    def __init__(self):
        super(TrViTB32AlexNet, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvitb32-tralex'

class UnMixerB16AlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(UnMixerB16AlexNet, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'unmixer-tralex'

class TrMixerB16AlexNet(UnMixerB16AlexNet):
    def __init__(self):
        super(TrMixerB16AlexNet, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trmixer-tralex'

class TrT2TT14AlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super(TrT2TT14AlexNet, self).__init__()
        self.source_model = 't2t_vit_t_14'
        self.source_pckg = 't2t'
        self.source_pretrain = True
        self.source_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = None
        self.savefile = 'trt2tv14_tralex' 

class LinAlexNet(DefaultArgs):
    def __init__(self):
        super(LinAlexNet, self).__init__()
        self.savefile = 'unalex-tralex_linear'
        self.metric = 'linear'

class PLSAlexNet(UnAlexNetAlexNet):
    def __init__(self):
        super().__init__()
        self.metric = 'PLS'
        self.params = [25]


########################################################################################
# target network = ResNet18
########################################################################################

class UnAlexNetResnet18(DefaultArgs):
    def __init__(self):
        super(UnAlexNetResnet18, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 'resnet18'
        self.source_pretrain = False
        self.target_pretrain = True
        self.target_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/resnet18/1/checkpoint.pth.tar' 
        self.source_layers = None
        self.target_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unalex-trres18'
        self.num_seeds = 1    
        self.params = [0.001,0.003,0.01,0.03,0.1,0.3]

class TrAlexNetResnet18(UnAlexNetResnet18):
    def __init__(self):
        super(TrAlexNetResnet18, self).__init__()
        self.source_pretrain = True
        self.savefile = 'tralex-trres18'

class UnVgg11Resnet18(UnAlexNetResnet18):
    def __init__(self):
        super(UnVgg11Resnet18, self).__init__()
        self.source_model = 'vgg11'
        self.source_layers = None
        self.source_types = (nn.ReLU)
        self.savefile = 'unvgg11-trres18'

class TrVgg11Resnet18(UnVgg11Resnet18):
    def __init__(self):
        super(TrVgg11Resnet18, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvgg11-trres18'

class UnResnet18Resnet18(UnAlexNetResnet18):
    def __init__(self):
        super(UnResnet18Resnet18, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unres18-trres18'

class TrResnet18Resnet18(UnResnet18Resnet18):
    def __init__(self):
        super(TrResnet18Resnet18, self).__init__()
        self.source_pretrain = True
        self.source_target_same = True
        self.savefile = 'trres18-trres18'

class TwoSeedsResnet18(UnResnet18Resnet18):
    def __init__(self):
        super(TwoSeedsResnet18, self).__init__()
        self.source_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/resnet18/2/checkpoint.pth.tar' 
        self.source_pretrain = True
        self.savefile = 'trres18_1-trres18_2'

class UnViTB32Resnet18(UnAlexNetResnet18):
    def __init__(self):
        super(UnViTB32Resnet18, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'unvitb32-trres18'

class TrViTB32Resnet18(UnViTB32Resnet18):
    def __init__(self):
        super(TrViTB32Resnet18, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvitb32-trres18'

class UnMixerB16Resnet18(UnAlexNetResnet18):
    def __init__(self):
        super(UnMixerB16Resnet18, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'unmixer-trres18'

class TrMixerB16Resnet18(UnMixerB16Resnet18):
    def __init__(self):
        super(TrMixerB16Resnet18, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trmixer-trres18'

class TrT2TT14Resnet18(UnAlexNetResnet18):
    def __init__(self):
        super(TrT2TT14Resnet18, self).__init__()
        self.source_model = 't2t_vit_t_14'
        self.source_pckg = 't2t'
        self.source_pretrain = True
        self.source_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = None
        self.savefile = 'trt2tv14_trres18' 


########################################################################################
# target network = Vgg11
########################################################################################

class UnAlexNetVgg11(DefaultArgs):
    def __init__(self):
        super(UnAlexNetVgg11, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 'vgg11'
        self.source_pretrain = False
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = (nn.ReLU)
        self.savefile = 'unalex-trvgg11'
        self.num_seeds = 1    
        self.params = [0.001,0.003,0.01,0.03,0.1,0.3]

        #for target unit experiments
        #self.savefolder = 'regress_unit_exp'

class TrAlexNetVgg11(UnAlexNetVgg11):
    def __init__(self):
        super(TrAlexNetVgg11, self).__init__()
        self.source_pretrain = True
        self.savefile = 'tralex-trvgg11'

class UnResnet18Vgg11(UnAlexNetVgg11):
    def __init__(self):
        super(UnResnet18Vgg11, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unres18-trvgg11'

class TrResnet18Vgg11(UnResnet18Vgg11):
    def __init__(self):
        super(TrResnet18Vgg11, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trres18-trvgg11'

class UnVgg11Vgg11(UnAlexNetVgg11):
    def __init__(self):
        super(UnVgg11Vgg11, self).__init__()
        self.source_model = 'vgg11'
        self.source_layers = None
        self.source_types = (nn.ReLU)
        self.savefile = 'unvgg11-trvgg11'

class TrVgg11Vgg11(UnVgg11Vgg11):
    def __init__(self):
        super(TrVgg11Vgg11, self).__init__()
        self.source_pretrain = True
        self.source_target_same = True
        self.savefile = 'trvgg11-trvgg11'

class TwoSeedsVgg11(UnVgg11Vgg11):
    def __init__(self):
        super(TwoSeedsVgg11, self).__init__()
        self.source_pretrain = True
        self.source_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/vgg11/2/checkpoint.pth.tar' 
        self.target_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/vgg11/1/checkpoint.pth.tar' 
        self.savefile = 'trvgg11_1-trvgg11_2'

class UnViTB32Vgg11(UnAlexNetVgg11):
    def __init__(self):
        super(UnViTB32Vgg11, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'unvitb32-trvgg11'

class TrViTB32Vgg11(UnViTB32Vgg11):
    def __init__(self):
        super(TrViTB32Vgg11, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvitb32-trvgg11'

class UnMixerB16Vgg11(UnAlexNetVgg11):
    def __init__(self):
        super(UnMixerB16Vgg11, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'unmixer-trvgg11'

class TrMixerB16Vgg11(UnMixerB16Vgg11):
    def __init__(self):
        super(TrMixerB16Vgg11, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trmixer-trvgg11'

class TrT2TT14Vgg11(UnAlexNetVgg11):
    def __init__(self):
        super(TrT2TT14Vgg11, self).__init__()
        self.source_model = 't2t_vit_t_14'
        self.source_pckg = 't2t'
        self.source_pretrain = True
        self.source_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = None
        self.savefile = 'trt2tv14_trvgg11' 

class TrVgg11Vgg11Unit(UnVgg11Vgg11):
    def __init__(self):
        super(TrVgg11Vgg11Unit, self).__init__()
        self.source_pretrain = True
        self.source_target_same = True
        self.source_layers = ['features.19']
        self.target_layers = ['features.19']
        self.source_types = None
        self.target_types = None
        #self.params = [1.0]
        self.savefolder = 'regress_unit_exp'
        self.savefile = 'trvgg11-trvgg11-unit'
        self.metric = 'PLS'
        self.params = None


########################################################################################
# target network = ViTB32
########################################################################################
class UnAlexNetViTB32(DefaultArgs):
    def __init__(self):
        super(UnAlexNetViTB32, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 'vit_base_patch32_224'
        self.target_pckg = 'timm'
        self.source_pretrain = False
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = None 
        self.savefile = 'unalex-trvitb32'
        self.num_seeds = 1    
        self.params = None

class TrAlexNetViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(TrAlexNetViTB32, self).__init__()
        self.source_pretrain = True
        self.savefile = 'tralex-trvitb32'

class UnResnet18ViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(UnResnet18ViTB32, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unres18-trvitb32'

class TrResnet18ViTB32(UnResnet18ViTB32):
    def __init__(self):
        super(TrResnet18ViTB32, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trres18-trvitb32'

class UnVgg11ViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(UnVgg11ViTB32, self).__init__()
        self.source_model = 'vgg11'
        self.source_layers = None
        self.source_types = (nn.ReLU)
        self.savefile = 'unvgg11-trvitb32'

class TrVgg11ViTB32(UnVgg11ViTB32):
    def __init__(self):
        super(TrVgg11ViTB32, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trvgg11-trvitb32'

class UnViTB32ViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(UnViTB32ViTB32, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'unvitb32-trvitb32'

class TrViTB32ViTB32(UnViTB32ViTB32):
    def __init__(self):
        super(TrViTB32ViTB32, self).__init__()
        self.source_pretrain = True
        self.source_target_same = True
        self.savefile = 'trvitb32-trvitb32'

class UnMixerB16ViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(UnMixerB16ViTB32, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'unmixer-trvitb32'

class TrMixerB16ViTB32(UnMixerB16ViTB32):
    def __init__(self):
        super(TrMixerB16ViTB32, self).__init__()
        self.source_pretrain = True
        self.savefile = 'trmixer-trvitb32'

class TrT2TT14ViTB32(UnAlexNetViTB32):
    def __init__(self):
        super(TrT2TT14ViTB32, self).__init__()
        self.source_model = 't2t_vit_t_14'
        self.source_pckg = 't2t'
        self.source_pretrain = True
        self.source_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = None
        self.savefile = 'trt2tv14_trvitb32' 

########################################################################################
# target network = Pytorch ViT_B_32 
########################################################################################
class TwoSeedsPytorchViTB32(DefaultArgs):
    def __init__(self):
        super(TwoSeedsPytorchViTB32, self).__init__()
        self.img_set = 'majaj-hong'
        self.source_model = 'vit_b_32'
        self.target_model = 'vit_b_32'
        self.source_pckg = 'pytorch'
        self.target_pckg = 'pytorch'
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/torchvision_recipe_vitb32/checkpoint.pth' 
        self.target_w_path = None
        self.source_layers = None
        self.target_layers = None
        self.source_types = None
        self.target_types = None
        self.srp_dim = None
        self.srp_bool = False
        self.params = None
        self.metric = 'CKA'
        self.savefolder = 'cka_normalize_crop'
        self.savefile = 'trvit_b_32_1-trvit_b_32_2'

########################################################################################
# target network = T2T_ViT_t_14
########################################################################################
class TrAlexNetT2TT14(DefaultArgs):
    def __init__(self):
        super(TrAlexNetT2TT14, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 't2t_vit_t_14'
        self.target_pckg = 't2t'
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = None 
        self.savefile = 'tralex-trt2tv14'
        self.num_seeds = 1    
        self.params = None

class TrResnet18T2TT14(TrAlexNetT2TT14):
    def __init__(self):
        super(TrResnet18T2TT14, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'trres18-trt2tv14'

class TrVgg11T2TT14(TrAlexNetT2TT14):
    def __init__(self):
        super(TrVgg11T2TT14, self).__init__()
        self.source_model = 'vgg11'
        self.source_layers = None
        self.source_types = (nn.ReLU)
        self.savefile = 'trvgg11-trt2tv14'

class TrViTB32T2TT14(TrAlexNetT2TT14):
    def __init__(self):
        super(TrViTB32T2TT14, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'trvitb32-trt2tv14'

class TrMixerB16T2TT14(TrAlexNetT2TT14):
    def __init__(self):
        super(TrMixerB16T2TT14, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'trmixer-trt2tv14'

class TrT2TT14T2TT14(TrAlexNetT2TT14):
    def __init__(self):
        super(TrT2TT14T2TT14, self).__init__()
        self.source_model = 't2t_vit_t_14'
        self.source_pckg = 't2t'
        self.source_pretrain = True
        self.source_layers = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                             ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
        self.source_types = None
        self.source_target_same = True
        self.savefile = 'trt2tv14_trt2tv14' 

########################################################################################
# target network = CORNet 
########################################################################################

class TwoSeedsCORNet(DefaultArgs):
    def __init__(self):
        super(TwoSeedsCORNet, self).__init__()
        self.img_set = 'majaj-hong'
        self.source_model = 'cornet_s'
        self.target_model = 'cornet_s'
        self.source_pckg = 'cornet'
        self.target_pckg = 'cornet'
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_w_path = '/cbcl/cbcl01/yenahan/brainmap/model_weights/cornet_s/latest_checkpoint.pth.tar' 
        self.target_w_path = None
        self.source_layers = None
        self.target_layers = None
        self.source_types = None
        self.target_types = None
        self.srp_dim = None
        self.srp_bool = False
        self.num_target_units = None
        self.params = None
        self.metric = 'CKA'
        self.savefolder = 'cka_normalize_crop'
        self.savefile = 'trcornet_s_1-trcornet_s_2'


########################################################################################
# testing CKA
class UnAlexNetViTB32CKA(DefaultArgs):
    def __init__(self):
        super(UnAlexNetViTB32CKA, self).__init__()
        self.source_model = 'alexnet'
        self.target_model = 'vgg11'#'vit_base_patch32_224'
        self.target_pckg = 'pytorch'
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None#['blocks.' + str(i) for i in range(12)]
        self.source_types = (nn.ReLU, nn.Linear)
        self.target_types = (nn.ReLU, nn.Linear)#None 
        self.srp_bool = False
        self.savefile = 'unalex-trvitb32-debug'
        self.num_seeds = 1    
        self.metric = 'ridge'#'CKA'
#        self.params = None
        self.params = [0.001, 0.005, 0.02, 0.1, 0.5, 2.0]#[0.001,0.003,0.01,0.03,0.1,0.3]

########################################################################################
# unit experiments, testing incomplete recordings
class TargetUnitsAlexNet(TwoSeedsAlexNet):
    def __init__(self):
        super(TargetUnitsAlexNet, self).__init__()
        self.metric = 'CKA'
        self.params = None
        self.srp_bool = False
        self.target_ratio = 100 
        self.num_target_units = None
        self.savefolder = 'cka_unit_exp'
        self.savefile = 'unit_alex_' + str(self.target_ratio).replace('.', '')

class TargetUnitVgg11AlexNet(TrVgg11AlexNet):
    def __init__(self):
        super(TargetUnitVgg11AlexNet, self).__init__()
        self.metric = 'CKA'
        self.params = None
        self.srp_bool = False
        self.target_ratio = 1
        self.num_target_units = None
        self.savefolder = 'cka_unit_exp'
        self.savefile = 'unit_vgg11_alex' + str(self.target_ratio).replace('.', '')

class TargetUnitResnet18AlexNet(TargetUnitVgg11AlexNet):
    def __init__(self):
        super(TargetUnitResnet18AlexNet, self).__init__()
        self.source_model = 'resnet18'
        self.source_layers = None
        self.source_types = (nn.ReLU, nn.Linear)
        self.savefile = 'unit_res18_alex' + str(self.target_ratio).replace('.', '')

class TargetUnitViTB32AlexNet(TargetUnitVgg11AlexNet):
    def __init__(self):
        super(TargetUnitViTB32AlexNet, self).__init__()
        self.source_model = 'vit_base_patch32_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) for i in range(12)]
        self.source_types = None
        self.savefile = 'unit_vitb32_alex' + str(self.target_ratio).replace('.', '')

class TargetUnitMixerB16AlexNet(TargetUnitVgg11AlexNet):
    def __init__(self):
        super(TargetUnitMixerB16AlexNet, self).__init__()
        self.source_model = 'mixer_b16_224'
        self.source_pckg = 'timm'
        self.source_layers = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                              ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
        self.source_types = None
        self.savefile = 'unit_mixer_alex' + str(self.target_ratio).replace('.', '')

class TargetUnitArgument(DefaultArgs):
    def __init__(self):
        super(TargetUnitArgument, self).__init__()
        self.img_set = 'majaj-hong'
        self.srp_dim = None
        self.params = None
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None
        self.source_types = None#(nn.ReLU, nn.Linear)
        self.target_types = None#(nn.ReLU, nn.Linear)
        self.srp_bool = False
        self.metric = 'CKA'
        self.num_target_units = None
        self.target_ratio = 1
        self.savefolder = 'cka_unit_exp3'
        self.savefile = None


########################################################################################
class CKAModelArgument(DefaultArgs):
    def __init__(self):
        super(CKAModelArgument, self).__init__()
        self.img_set = 'imagenet'
        self.srp_dim = None
        self.params = None
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None
        self.source_types = None#(nn.ReLU, nn.Linear)
        self.target_types = None#(nn.ReLU, nn.Linear)
        self.srp_bool = False
        self.num_target_units = None
        self.metric = 'CKA'
        self.savefolder = 'motif'
        self.savefile = None
 
class CKAUntrainedModelArgument(DefaultArgs):
    def __init__(self):
        super(CKAUntrainedModelArgument, self).__init__()
        self.img_set = 'imagenet'
        self.srp_dim = None
        self.params = None
        self.source_pretrain = False
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None
        self.source_types = None#(nn.ReLU, nn.Linear)
        self.target_types = None#(nn.ReLU, nn.Linear)
        self.srp_bool = False
        self.metric = 'CKA'
        self.savefolder = 'motif'
        self.savefile = None

class CKAUnitModelArgument(CKAModelArgument):
    def __init__(self):
        super(CKAUnitModelArgument, self).__init__()
        self.num_target_units = None
 
class RidgeModelArgument(DefaultArgs):
    def __init__(self):
        super(RidgeModelArgument, self).__init__()
        self.img_set = 'imagenet'
        self.srp_dim = 6862 # majaj-hong:6917, imagenet:6862, freeman-ziemba:4202 
        self.params = [1.0]
        self.source_pretrain = True
        self.target_pretrain = True
        self.source_layers = None
        self.target_layers = None
        self.source_types = None#(nn.ReLU, nn.Linear)
        self.target_types = None#(nn.ReLU, nn.Linear)
        self.srp_bool = True
        self.metric = 'ridge'
        self.savefolder = 'motif/ridge'
        self.savefile = None
 

