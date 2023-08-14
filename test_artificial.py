import os
from os import listdir
from os.path import isfile, join
import json
import argparse
from importlib import import_module
import faulthandler
import gc
import pickle
import time
import collections

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image

import h5py
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr

# import open_clip
import timm
from timm.data import transforms_factory 

import artificial_configs
# import cca_core
from T2T_ViT import models as t2t_models
from T2T_ViT.utils import load_for_transfer_learning 
from CORnet import cornet as cornet


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.layer_count = {}
        self._features = {}

        for layer_name, layer in model.named_modules():
            if layer_name in self.layers:
                layer.register_forward_hook(self.save_features_hook(layer_name))
 
    def save_features_hook(self, layer_name):
        def fn(_, __, output):
            if layer_name in self._features: # some layers are not named but reused in forward fn
                self.layer_count[layer_name] = self.layer_count.get(layer_name,0) + 1
                self._features[layer_name + '-' + str(self.layer_count[layer_name])] = output.flatten(start_dim=1)     
            else:    
                self._features[layer_name] = output.flatten(start_dim=1).detach().cpu()
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features

def get_saved_acts(pckg, model, weight_path, img_set, trained=True, num_imgs=None, svd=False):
    save_base_dir = '/cbcl/cbcl01/yenahan/brainmap/saved_activations'
    if weight_path:
        weight_path = weight_path.replace('.', '')
        model_id = ''.join(weight_path.split('/')[6:])
        save_id = f'{model_id}-{img_set}'
    elif trained:
        save_id = f'{pckg}-{model}-{img_set}' 
    else:
        save_id = f'{pckg}-untrained_{model}-{img_set}' 
    
    if img_set == 'majaj-hong' and num_imgs != 3200:  # if not the default num_imgs
        save_id += f'{num_imgs}'
    elif img_set == 'imagenet' and num_imgs != 3000:  # if not the default num_imgs
        save_id += f'{num_imgs}'

    if svd:
        save_id += '-svd'

    saved_file = f'{save_base_dir}/{save_id}.pkl'
    if os.path.isfile(saved_file):
        with open(saved_file, 'rb') as f:
            activations = pickle.load(f)
        return activations
    else:
        return False

def save_acts(pckg, model, img_set, acts, trained=True, weight_path = None,
        num_imgs=None, svd=False):
    save_base_dir = '/cbcl/cbcl01/yenahan/brainmap/saved_activations'
    if weight_path:
        weight_path = weight_path.replace('.', '')
        model_id = ''.join(weight_path.split('/')[6:])
        save_id = f'{model_id}-{img_set}'
    elif trained:
        save_id = f'{pckg}-{model}-{img_set}'   
    else:
        save_id = f'{pckg}-untrained_{model}-{img_set}'   

    if img_set == 'majaj-hong' and num_imgs != 3200:  # if not the default num_imgs
        save_id += f'{num_imgs}'
    elif img_set == 'imagenet' and num_imgs != 3000:  # if not the default num_imgs
        save_id += f'{num_imgs}'
    
    if svd:
        save_id += '-svd'

    saved_file = f'{save_base_dir}/{save_id}.pkl'
    if os.path.isfile(saved_file):
        return
    else:
        print(f'saving activations to {saved_file}')
        with open(saved_file, 'wb') as f:
            pickle.dump(acts, f, protocol=4)


def get_layers(model_name, model, layers, layer_types):
    assert not (layers and layer_types)
    if layers:
        return layers
    if layer_types:
        return get_layer_by_type(model, layer_types)

    # if layers are not specified either by names or types use pre-defined default layers
    pre_defined_layers = {}
    pre_defined_layers['t2t_vit_t_14'] = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                                     ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(14)]
    pre_defined_layers['t2t_vit_t_19'] = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                                     ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(19)]
    pre_defined_layers['t2t_vit_7'] = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                                     ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(7)]
    pre_defined_layers['t2t_vit_10'] = ['tokens_to_token.attention' + str(i+1) for i in range(2)] + \
                                     ['tokens_to_token.project'] + ['blocks.' + str(i) for i in range(10)]
    pre_defined_layers['vit_b_32'] = ['encoder.layers.encoder_layer_' + str(i) for i in range(12)]
    pre_defined_layers['vit_b_16'] = ['encoder.layers.encoder_layer_' + str(i) for i in range(12)]
    pre_defined_layers['vit_l_32'] = ['encoder.layers.encoder_layer_' + str(i) for i in range(24)]
    pre_defined_layers['vit_l_16'] = ['encoder.layers.encoder_layer_' + str(i) for i in range(24)]
    pre_defined_layers['swin_t'] = ['features.1.0', 'features.1.1', 'features.3.0', 'features.3.1'] + \
            ['features.5.' + str(i) for i in range(6)] + ['features.7.0', 'features.7.1']
    pre_defined_layers['swin_s'] = ['features.1.0', 'features.1.1', 'features.3.0', 'features.3.1'] + \
            ['features.5.' + str(i) for i in range(18)] + ['features.7.0', 'features.7.1']
    pre_defined_layers['swin_b'] = ['features.1.0', 'features.1.1', 'features.3.0', 'features.3.1'] + \
            ['features.5.' + str(i) for i in range(18)] + ['features.7.0', 'features.7.1']
    pre_defined_layers['visformer_small'] = ['stage1.' + str(i) for i in range(7)] + \
            ['stage2.' + str(i) for i in range(4)] + ['stage3.' + str(i) for i in range(4)] + ['global_pool']
    pre_defined_layers['twins_pcpvt_small'] = ['blocks.0.' + str(i) for i in range(3)] + \
            ['blocks.1.' + str(i) for i in range(4)] + ['blocks.2.' + str(i) for i in range(6)] +\
            ['blocks.3.' + str(i) for i in range(3)]
    pre_defined_layers['twins_svt_small'] = ['blocks.0.' + str(i) for i in range(2)] + \
            ['blocks.1.' + str(i) for i in range(2)] + ['blocks.2.' + str(i) for i in range(10)] +\
            ['blocks.3.' + str(i) for i in range(4)]
    pre_defined_layers['mobilenet_v2'] = ['features.' + str(i) for i in range(12)] + ['classifier.0', 'classifier.1']
    pre_defined_layers['squeezenet1_0'] = ['features.' + str(i) for i in range(13)]
    pre_defined_layers['mixer_b16_224'] = ['blocks.' + str(i) + '.mlp_tokens.fc2' for i in range(12)] + \
                                          ['blocks.' + str(i) + '.mlp_channels.fc2' for i in range(12)] 
    
    pre_defined_layers['openclip-ViT-B-32'] = ['transformer.resblocks.' + str(i) for i in range(12)]
    pre_defined_layers['openclip-ViT-L-14'] = ['transformer.resblocks.' + str(i) for i in range(24)]
    pre_defined_layers['openclip-convnext_base_w'] = ['trunk.stem'] + \
            ['trunk.stages.0.blocks.' + str(i) for i in range(3)] + \
            ['trunk.stages.1.blocks.' + str(i) for i in range(3)] + \
            ['trunk.stages.2.blocks.' + str(i) for i in range(0,27,3)] + \
            ['trunk.stages.3.blocks.' + str(i) for i in range(3)] + \
            ['trunk', 'head']
 
    pre_defined_layers['openclip-convnext_large_d'] = ['trunk.stem'] + \
            ['trunk.stages.0.blocks.' + str(i) for i in range(3)] + \
            ['trunk.stages.1.blocks.' + str(i) for i in range(3)] + \
            ['trunk.stages.2.blocks.' + str(i) for i in range(0,27,3)] + \
            ['trunk.stages.3.blocks.' + str(i) for i in range(3)] + \
            ['trunk', 'head']
 
    if model_name == 'alexnet':
        pre_defined_layers[model_name] = get_layer_by_type(model, (nn.ReLU, nn.Linear))
    elif 'vgg' in model_name:
        pre_defined_layers[model_name] = get_layer_by_type(model, (nn.ReLU))
    elif 'resnet' in model_name:
        pre_defined_layers[model_name] = get_layer_by_type(model, (nn.ReLU, nn.Linear))
    elif model_name == 'densenet121':
        all_layers = get_layer_by_type(model, (nn.ReLU))
        pre_defined_layers[model_name] = all_layers[::4]
    elif 'cornet' in model_name:
        pre_defined_layers[model_name] = get_layer_by_type(model, (nn.ReLU, nn.AdaptiveAvgPool2d))

    return pre_defined_layers[model_name] 

def get_layer_by_type(model, layer_types):
    layer_sel = []
    for name, layer in model.named_modules():
        if isinstance(layer, layer_types):
            layer_sel.append(name)
    return layer_sel

def get_images(image_type, transform, num_images=1000):
    if image_type in ['majaj-hong', 'freeman-ziemba', 'freeman-ziemba-target8-source4']:
        image_paths = get_brainscore_imgs(image_type)
    elif image_type == 'imagenet':
        image_paths = get_imagenet_val(num_images)
    elif image_type == 'laion':
        image_paths = get_laion_imgs(num_images)
    
    img_list = []
    for image in image_paths:
        img = Image.open(image).convert('RGB')
        img = transform(img)
        img_list.append(img)
    inputs = torch.stack(img_list)
    return inputs
 
def get_imagenet_val(num_images):
    # copied from brainscore codebase
    num_classes = 1000
    num_images_per_class = (num_images - 1) // num_classes
    base_indices = np.arange(num_images_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)
    for i in range((num_images - 1) % num_classes + 1):
        indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

    framework_home = '/cbcl/cbcl01/yenahan/.model-tools/'
    imagenet_filepath = os.path.join(framework_home, 'imagenet2012.hdf5')
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    #if not os.path.isfile(imagenet_filepath):
    #    os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
    #    s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    return filepaths

def get_brainscore_imgs(image_type='majaj-hong'):
    base_dir = '/cbcl/cbcl01/yenahan/brainmap/imgs'
    if image_type == 'majaj-hong':
        img_path = f'{base_dir}/image_dicarlo_hvm-public/'    
    elif image_type == 'freeman-ziemba':
        img_path = f'{base_dir}/image_movshon_FreemanZiemba2013_aperture-public/'
    elif image_type == 'freeman-ziemba-target8-source4':
        img_path = f'{base_dir}/movshon.FreemanZiemba2013.aperture-public--target8--source4/'
    return [join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f))]


def get_laion_imgs(num_images):
    img_path = '/cbcl/cbcl01/yenahan/brainmap/dataset/laion400m'

    filepaths = []
    cnt = 0
    index = 0
    while cnt < num_images:
        if os.path.exists(f'{img_path}/{index}.jpg'):
            filepaths.append(f'{img_path}/{index}.jpg')
            cnt += 1
        index += 1
    assert cnt == num_images
    return filepaths


def get_image_transform(model_name, package, resize=224):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    IMAGENET_INCEPTION_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_INCEPTION_STD = [0.5, 0.5, 0.5]    
    OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
    
    if package == 'timm' and model_name in ['vit_base_patch32_224', 'mixer_b16_224']:
        print(model_name, ': vit transform')
        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD
    elif package == 'open_clip':
        print(model_name, ': clip transform')
        mean = OPENAI_DATASET_MEAN
        std = OPENAI_DATASET_STD

    else:
        print(model_name, ': standard transform')
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    if model_name in ['convnext_base_w', 'convnext_large_d']:
        input_dim = 256
    else:
        input_dim = 224

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([transforms.Resize(resize),
                           transforms.CenterCrop(input_dim),
                           transforms.ToTensor(),
                                   normalize])
#        transform = transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                        normalize])
    return transform

def get_model(model_name, pretrain, package=None, weight_path=None):
    # if specified to use saved weights
    # only works for pytorch vision models
    if weight_path and package == 'pytorch':
        model = models.__dict__[model_name]()
        checkpoint = torch.load(weight_path)

        if model_name == 'vit_b_32':
            state_dict =checkpoint['model']
        else:
            state_dict =checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # removing ‘.moldule’ from key
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)

    else:
        if package == 'timm':
            model = timm.create_model(model_name, pretrained=pretrain)
        elif package == 'pytorch':
            # for recent models, access models via PyTorch Hub
            # run swin with conda env pytorch1_11 (or nightly version) 
            if 'swin' in model_name:
                assert pretrain, 'untrained swin models are selected'
                model = torch.hub.load('pytorch/vision', model_name, weights='IMAGENET1K_V1')
            else:
                model = models.__dict__[model_name](pretrained=pretrain)
        elif package == 't2t':
            base_dir = '/cbcl/cbcl01/yenahan/brainmap/T2T_ViT/model_weights'
            if model_name == 't2t_vit_t_14':
                w = f'{base_dir}/81.7_T2T_ViTt_14.pth.tar'
            elif model_name == 't2t_vit_t_19':
                w = f'{base_dir}/82.4_T2T_ViTt_19.pth.tar'
            elif model_name == 't2t_vit_14':
                w = f'{base_dir}/81.5_T2T_ViT_14.pth.tar'
            elif model_name == 't2t_vit_7':
                w =  f'{base_dir}/71.7_T2T_ViT_7.pth.tar'
            elif model_name == 't2t_vit_10':
                w = f'{base_dir}/75.2_T2T_ViT_10.pth.tar'
            model = t2t_models.__dict__[model_name]()
            load_for_transfer_learning(model, w, use_ema=True, strict=False, num_classes=1000)
        elif package == 'cornet':
            map_location = 'cpu'
            model = getattr(cornet, f'{model_name}')
            if model_name == 'cornet_r':
                model = model(pretrained=(pretrain and not weight_path), map_location=map_location, times=5)
            else:
                model = model(pretrained=(pretrain and not weight_path), map_location=map_location)
            model = model.module  # remove DataParallel

            if weight_path:
                checkpoint = torch.load(weight_path)
                state_dict =checkpoint['state_dict']
                
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "") # removing ‘.moldule’ from key
                    new_state_dict[name]=v
                model.load_state_dict(new_state_dict)
        elif package == 'open_clip':
            # use laion2b as pre-training datasets 
            if model_name == 'convnext_base_w':
                model = open_clip.create_model('convnext_base_w', 
                        pretrained='laion2b_s13b_b82k')
            elif model_name == 'convnext_large_d':
                model = open_clip.create_model('convnext_large_d', 
                         pretrained='laion2b_s26b_b102k_augreg')
            elif model_name == 'ViT-B-32':
                model = open_clip.create_model('ViT-B-32', 
                        pretrained='laion2b_s34b_b79k')
            elif model_name == 'ViT-L-14':
                model = open_clip.create_model('ViT-L-14',
                        pretrained='laion2b_s32b_b82k')



    return model


def get_activations(pckg, model_name, weight_path, trained, layers, layer_types,
        img_set, num_imgs, apply_svd=False):
    # check if extracted activations are saved
    saved_acts = get_saved_acts(pckg, model_name, weight_path, img_set, 
            trained, num_imgs, apply_svd)

    model = get_model(model_name, trained, pckg, weight_path)
    layers = get_layers(model_name, model, layers, layer_types)
    print('layers: ', layers)

    if saved_acts:
        print('using saved activations')
        return saved_acts, layers
   
    # if activations are not saved
    device = torch.device('cpu')
    image_transform = get_image_transform(model_name, pckg)
    image_inputs = get_images(img_set, image_transform, num_imgs)
    image_inputs = image_inputs.to(device)
    print('image_input shape: ', image_inputs.shape)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        feature_extractor = FeatureExtractor(model, layers)
        features = feature_extractor(image_inputs)
 
    if apply_svd:
        features = svd(features)

    # save activations
    save_acts(pckg, model_name, img_set, features, trained, weight_path, num_imgs, apply_svd)

    return features, layers

def svd(acts):
    svd_acts = {}
    for layer in acts:
        layer_acts = acts[layer]

        layer_acts = np.array(layer_acts)
        # Mean subtract activations
        cacts = layer_acts - np.mean(layer_acts, axis=0, keepdims=True)

        # Perform SVD
        U, s, V = np.linalg.svd(cacts, full_matrices=False)

        svd_acts[layer] = {'U': U, 's': s, 'V': V}
    return svd_acts 
        
def srp(out, srp_dim=1000):
    if srp_dim:
        n_components = srp_dim
    else:
        num_imgs = out[out.keys()[0]].shape[0]
        n_components = johnson_lindenstrauss_min_dim(num_imgs, eps=0.1)

    srp_acts = {}
    for layer in out:
        acts = out[layer]
        proj = SparseRandomProjection(n_components, random_state=0)
        flattened = acts.reshape(acts.shape[0], -1)
        srp_acts[layer] = proj.fit_transform(flattened.cpu())
    return srp_acts


# score computations adapted from https://github.com/ColinConwell/DeepMouseTrap/
pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')

def pearson_r_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    
    return(pearsonr_vec(y_true_, y_pred_)[0])

def pearson_r2_score(y_true, y_pred, multioutput=None):
    return(pearson_r_score(y_true, y_pred)**2)

def get_predicted_values(y_true, y_pred, transform = None, multioutput = None):
    if transform == None:
        return(y_pred)

scoring_options = {'r2': r2_score, 'pearson_r': pearson_r_score, 'pearson_r2': pearson_r2_score,
                   'explained_variance': explained_variance_score, 'predicted_values': get_predicted_values}

def get_scoring_options():
    return scoring_options

def score_func(y_true, y_pred, score_type='pearson_r'):
    if not isinstance(score_type, list):
        return(scoring_options[score_type](y_true, y_pred, multioutput='raw_values'))
    
    if isinstance(score_type, list):
        scoring_dict = {}
        for score_type_i in score_type:
            scoring_dict[score_type_i] = scoring_options[score_type_i](y_true, y_pred, multioutput='raw_values')
        
    return(scoring_dict)

def CKA(target, features):
    X, Y = features.numpy(), target
    
    X_n = X
    Y_n = Y
    m = X_n.shape[0]
    K = np.matmul(X_n, X_n.T)
    L = np.matmul(Y_n, Y_n.T)
    H = np.identity(m) - np.ones((m,m)) / m
    K_c = np.matmul(np.matmul(H, K), H)
    L_c = np.matmul(np.matmul(H, L), H)
    HSIC_KL = np.dot(K_c.flatten(), L_c.flatten()) / ((m-1)**2)
    HSIC_KK = np.dot(K_c.flatten(), K_c.flatten()) / ((m-1)**2)
    HSIC_LL = np.dot(L_c.flatten(), L_c.flatten()) / ((m-1)**2)
    return HSIC_KL / np.sqrt(HSIC_KK * HSIC_LL)


def regression(target, features, regress_type='PLS', param=0.01, cv_splits = 10, score_type='pearson_r'):    
    X,y = features, target
    #print('X_shape, y_shape', X.shape, y.shape)
    
    if regress_type == 'PLS':
        # following default parameters used in brainscore
        # default = 25
        regression = PLSRegression(n_components=25, scale=False)
    elif regress_type == 'linear':
        regression = LinearRegression(normalize=False)
    kfolds = KFold(cv_splits, shuffle=False).split(np.arange(y.shape[0]))
    
    start = time.time()
    y_pred = np.zeros((y.shape[0],y.shape[1]))
    y_train_all = np.zeros((y.shape[0],y.shape[1]))
    #print('y_pred_shape: ', y_pred.shape)
    for train_indices, test_indices in kfolds:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        if regress_type == 'ridge':
            # standardize X and demean y
            X_mean = X_train.mean(axis=0)
            y_mean = y_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_train_n = (X_train - X_mean) / X_std
            X_test_n = (X_test - X_mean) / X_std
            y_train_n = y_train - y_mean 
            XX = np.dot(X_train_n.T, X_train_n) / X_train_n.shape[0]
            XY = np.dot(X_train_n.T, y_train_n) / X_train_n.shape[0]
            beta = np.linalg.solve(XX + param*np.identity(XX.shape[0]), XY)
            y_pred[test_indices] = np.dot(X_test_n, beta) + y_mean
            y_train_all[train_indices] = np.dot(X_train_n, beta) + y_mean
        else:
            regression = regression.fit(X_train, y_train)
            y_pred[test_indices] = regression.predict(X_test)
            y_train_all[train_indices] = regression.predict(X_train)

    end = time.time()
    print('elapsed time for regression', end-start)
    
    return score_func(y, y_pred, score_type), score_func(y, y_train_all, score_type)

def cca(acts1, acts2, gpu_index=None):
    # acts1: target, acts2: features
    U2 = acts2['U']
    s2 = acts2['s']
    V2 = acts2['V']

    def explain_fraction(s):
        i = 0
        s_large_sum = s[0]
        s_sum = np.sum(s)
        while s_large_sum / s_sum < 0.98:
            i += 1
            s_large_sum += s[i]
        return i

    acts1 = np.array(acts1)
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=0, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)

    i1 = explain_fraction(s1)
    i2 = explain_fraction(s2)

    # X is transposed compared to google cca codebase
    svacts1 = np.dot(s1[:i1]*np.eye(i1), U1.T[:i1])
    svacts2 = np.dot(s2[:i2]*np.eye(i2), U2.T[:i2])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    result = {'cca_coef': svcca_results['cca_coef1'],
            'dimension': {'target': i1, 'feature': i2}}
    return result 


def preprocess_regression(X_train, X_test, Y_train, Y_test):
    # return XX and XY
    # standardize X and demean Y
    X_mean = torch.mean(X_train, dim=0)
    Y_mean = torch.mean(Y_train, dim=0)
    X_std = torch.std(X_train, dim=0)
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std
    Y_train_n = Y_train - Y_mean
    XX = torch.matmul(torch.t(X_train_n), X_train_n) / X_train_n.shape[0]
    XY = torch.matmul(torch.t(X_train_n), Y_train_n) / X_train_n.shape[0]
    return XX, XY, X_test_n, X_train_n, Y_mean

def nested_cv_regression(target, features, regress_type='ridge', params=[1.0],
        outer_splits=5, inner_splits=5, score_type='pearson_r', gpu_index=1):
    assert regress_type == 'cv_ridge', 'gpu regression only supports ridge regression'
    if gpu_index == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_index}')
    features, target = (torch.tensor(features, dtype=torch.float64), 
            torch.tensor(target, dtype=torch.float64))
    X,Y = features.to(device), target.to(device)
    #print('X_shape, y_shape', X.shape, Y.shape)
    # X = N*P (N: number of images, P: feature dimension)
    # Y = N*T (T: number of target units)

    # Y_pred = N*T, Y_train_all = N*T
    # best_params_all = num_outer_splits*T
    Y_pred = torch.zeros(Y.shape[0], Y.shape[1], dtype=torch.float64).to(device)
    Y_train_all = torch.zeros(Y.shape[0], Y.shape[1], dtype=torch.float64).to(device)
    best_params_all = np.zeros((outer_splits, Y.shape[1]))

    outer_kfolds = KFold(outer_splits, shuffle=True).split(np.arange(Y.shape[0]))
    for kfold_i, (train_indices, test_indices) in enumerate(outer_kfolds):
        # inner_Y_pred = L*IN*T (L: number of params, IN: number of training images for inner loop)
        inner_Y_pred = torch.zeros(len(params), len(train_indices), 
                Y.shape[1], dtype=torch.float64).to(device)
        
        # inner_kfolds = KFold(inner_splits, shuffle=False).split(np.arange(len(train_indices)))
        inner_kfolds = KFold(inner_splits, shuffle=True).split(train_indices)
        
        for inner_train_indices, inner_test_indices in inner_kfolds:
            global_inner_train_indices = train_indices[inner_train_indices]
            global_inner_test_indices = train_indices[inner_test_indices]
            X_train = X[global_inner_train_indices, :]
            X_test = X[global_inner_test_indices, :]
            Y_train = Y[global_inner_train_indices, :]
            Y_test = Y[global_inner_test_indices, :]

            XX, XY, X_test_n, X_train_n, Y_mean = preprocess_regression(X_train, X_test,
                    Y_train, Y_test)
            if gpu_index == 'cpu':
                XX = XX.to(device)
                XY = XY.to(device)
                X_test_n = X_test_n.to(device)
                X_train_n = X_train_n.to(device)
                Y_mean = Y_mean.to(device)
            identity = torch.eye(torch.tensor(XX.shape[0])).to(device)
            
            for param_i, param in enumerate(params):
                beta = torch.linalg.solve(XX + param*identity, XY)
                inner_Y_pred[param_i, inner_test_indices] = torch.matmul(X_test_n, beta) + Y_mean
    
        # compute correlation score
        scores = torch.zeros(len(params), Y.shape[1])
        for param_i, param in enumerate(params):
            scores[param_i] = gpu_score_func(Y[train_indices], inner_Y_pred[param_i])

        # choose param s.t. score is max for each target
        # num_targets
        max_param_i = torch.argmax(scores, dim=0)
        # best_params = np.array(params)[max_param_i]
        best_params_all[kfold_i, :] = max_param_i

        # compute beta using optimal param
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        XX, XY, X_test_n, X_train_n, Y_mean = preprocess_regression(X_train, X_test,
                Y_train, Y_test)
        if gpu_index == 'cpu':
            XX = XX.to(device)
            XY = XY.to(device)
            X_test_n = X_test_n.to(device)
            X_train_n = X_train_n.to(device)
            Y_mean = Y_mean.to(device)
        identity = torch.eye(torch.tensor(XX.shape[0])).to(device)

        beta = torch.zeros(XX.shape[0], XY.shape[1], dtype=torch.float64).to(device)
        for param_i, param in enumerate(params):
            beta_indices = [i for i in range(max_param_i.shape[0]) if max_param_i[i] == param_i]
            # XY slicing, only units whose optimal param is current param
            beta[:, beta_indices] = torch.linalg.solve(XX + param*identity,
                    XY[:, beta_indices])

        # prediction
        Y_pred[test_indices] = torch.matmul(X_test_n, beta) + Y_mean
        Y_train_all[train_indices] = Y_train_all[train_indices] + \
                torch.matmul(X_train_n, beta) + Y_mean

    # compute average for training split predictions
    Y_train_all = Y_train_all / outer_splits
    
    return gpu_score_func(Y, Y_pred), gpu_score_func(Y, Y_train_all), best_params_all
        
def gpu_score_func(Y, Y_pred):
    # compute pearson r correlation
    Y_mean = torch.mean(Y, dim=0)
    Y_pred_mean = torch.mean(Y_pred, dim=0)
    Y_std = torch.std(Y, dim=0)
    Y_pred_std = torch.std(Y_pred, dim=0)
    Y_dm = Y - Y_mean
    Y_pred_dm = Y_pred - Y_pred_mean
    cov = torch.mean(torch.mul(Y_dm, Y_pred_dm), dim=0)
    std = torch.mul(Y_std, Y_pred_std)
    # avoid division by 0
    std_inv = torch.clone(std)
    std_inv[std > 0] = 1.0 / std[std > 0]
    score = cov * std_inv
    return score

def subsample_targets(target_features, target_layer, num_target_units, target_ratio):
    target_var = target_features[target_layer].var(axis=0)
    nonzero_var = target_var != 0
    nonzero_indices = np.arange(target_features[target_layer].shape[1])[nonzero_var]
    if num_target_units:
        # filter out units with var=0, and sample num_target_units 
        indices = np.random.choice(nonzero_indices, num_target_units)
        target = target_features[target_layer].numpy()[:, indices]
        num_record_units = num_target_units
    elif target_ratio:
        # filter out units with var=0, and sample target_ratio 
        num_record_units = int(len(nonzero_indices) * target_ratio / 100)
        indices = np.random.choice(nonzero_indices, num_record_units)
        target = target_features[target_layer].numpy()[:, indices]
    else:  # include all units
        target = target_features[target_layer].numpy()
        num_record_units = 'all'
    print(f'num_target_units of {target_layer}: {num_record_units}')
    return target

def check_arguments(args):
    if args.source_target_same:
        assert (args.source_model == args.target_model and 
                (args.source_pretrain == args.target_pretrain)),\
                        "Cannot use same models. Architecture type or pretraining differs"

    assert not (args.num_target_units and args.target_ratio),\
            "Cannot use both num_target_units and target_ratio"

    assert args.metric in ['CKA', 'cv_ridge'],\
            "Current version only supports CKA or cv_ridge"

def main(args): 
    start = time.time()
    print('enters main function')
    faulthandler.enable()

    check_arguments(args)

    print('source activations')
    source_features, source_layers = get_activations(args.source_pckg, args.source_model,
            args.source_w_path, args.source_pretrain, args.source_layers, args.source_types,
            args.img_set, args.num_imgs, args.metric == 'CCA')
    print('target activations')
    target_features, target_layers = get_activations(args.target_pckg, args.target_model,
            args.target_w_path, args.target_pretrain,  args.target_layers, args.target_types,
            args.img_set, args.num_imgs) 
    
    args.source_layers = source_layers    
    args.target_layers = target_layers

    all_scores = {}
    for seed_i in range(args.num_seeds):
        if args.srp_bool:
            acts = srp(source_features, args.srp_dim)
        
            for layer in source_features:
                print(layer, 'feature_shape:', source_features[layer].shape,
                        'srp_shape:', acts[layer].shape)
            del source_features
            gc.collect()
        elif args.metric == 'CCA':
            acts = source_features
            for layer in source_features:
                print(layer, 'svd_feature_shape:', source_features[layer]['V'].shape)
        else:
            acts = source_features
            for layer in source_features:
                print(layer, 'feature_shape:', source_features[layer].shape)

        for target_layer in target_layers:
            target = subsample_targets(target_features, target_layer, args.num_target_units,
                        args.target_ratio)
            
            for source_layer in source_layers:
                scores_for_params = {'test': [], 'train': []}

                if args.metric == 'CKA':
                    score, train_score = CKA(target, acts[source_layer]), 0
                    median_score = np.ma.median(score)
                    median_train_score = np.ma.median(train_score)
                   
                    test_score = {'val': score, 'median': median_score}
                    train_score = {'val': train_score, 'median': median_train_score}
    
                elif args.metric == 'CCA':
                    score_result, train_score = cca(target, acts[source_layer]), 0
                    score = score_result['cca_coef']
                    svd_dim = score_result['dimension']
                    median_score = np.mean(score)
                    median_train_score = np.ma.median(train_score)
 
                    test_score = {'val': score.tolist(), 'median': median_score}
                    train_score = {'val': train_score, 'median': median_train_score}

                    print(f"svd_dimension target: {svd_dim['target']},"\
                          f" feature: {svd_dim['feature']}")
 
                elif args.metric == 'cv_ridge':
                    score, train_score, best_params = nested_cv_regression(target, acts[source_layer],
                            regress_type=args.metric, params=args.params,
                            outer_splits=args.outer_cv_splits,
                            inner_splits=args.inner_cv_splits,
                            gpu_index=args.gpu_index)
                    score, train_score = score.to('cpu'), train_score.to('cpu')

                    # overall count per param index
                    params_count = collections.defaultdict(int)
                    for param in best_params.flatten().tolist():
                        params_count[param] += 1
                    param_summary = {'best': best_params.tolist(),'counter': params_count}

                    median_score = np.ma.median(score)
                    median_train_score = np.ma.median(train_score)

                    test_score = {'val': score.detach().numpy().tolist(), 'median': median_score}
                    train_score = {'val': train_score.detach().numpy().tolist(), 'median': median_train_score}

                pair_id = source_layer + '-' + target_layer
                print(pair_id + ' median test score: ', median_score, 
                        'median train score: ', median_train_score)

                if pair_id not in all_scores:
                    all_scores[pair_id] = []
                if args.metric == 'CKA':
                    all_scores[pair_id].append({'seed_i': seed_i, 
                        'test_score': test_score, 'train_score': train_score})
                elif args.metric == 'CCA':
                    all_scores[pair_id].append({'seed_i': seed_i, 
                        'dimension': svd_dim,
                        'test_score': test_score, 'train_score': train_score})
                elif args.metric == 'cv_ridge':
                    all_scores[pair_id].append({'seed_i': seed_i, 
                        'max_param': param_summary,
                        'test_score': test_score, 'train_score': train_score})
           
            if args.target_ratio and 'num_record_units' not in all_scores:
                all_scores['num_record_units'] = {target_layer: num_record_units}
            elif args.target_ratio:
                all_scores['num_record_units'][target_layer] = num_record_units
            
    if args.source_types:
        args.source_types = str(args.source_types)
    if args.target_types:
        args.target_types = str(args.target_types)
    all_scores['parameters'] = args.__dict__

    if not args.savefile:
        args.savefile = f'{args.source_model}-{args.target_model}'
    for i in range(0, 1000):
        if args.savefolder:
            saveat = args.savefolder + '/' + args.savefile
        else:
            saveat = args.savefile
        p = '/cbcl/cbcl01/yenahan/brainmap/artificial_score/' + saveat + '.json'
        if i > 0:
            p += f'.{i}'
        if not os.path.isfile(p):
            print(f'saving to {p}')
            with open(p, 'w') as f:
                json.dump(all_scores, f, indent=4)
            break
    end = time.time()
    print(f'elapsed time: {end-start}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', required=True)
    parser.add_argument('--metric')
    parser.add_argument('--no-params', type=bool)
    parser.add_argument('--srp_bool', type=bool)
    parser.add_argument('--srp_dim', type=int)
    parser.add_argument('--num_target_units', type=int)
    parser.add_argument('--target_ratio', type=int)
    parser.add_argument('--img_set')
    parser.add_argument('--num_imgs', type=int)
    parser.add_argument('--savefolder')
    parser.add_argument('--savefile')
    parser.add_argument('--source_model')
    parser.add_argument('--target_model')
    parser.add_argument('--source_pckg')
    parser.add_argument('--target_pckg')
    parser.add_argument('--source_w_path')
    parser.add_argument('--gpu_index')
    
    args = parser.parse_args()
    print('argument', args.__dict__)
    module = import_module('artificial_configs')
    config = getattr(module, args.config_name)()
    for std_argument, value in args.__dict__.items():
        if value:
            if std_argument == 'no-params' and args.no_params:
                config.params = None
            if std_argument == 'metric' and args.metric == 'ridge_1':
                config.metric = 'ridge'
                config.params = [1.0]
            elif std_argument == 'metric' and args.metric == 'ridge_2':
                config.metric = 'ridge'
                config.params = [2.0]
            elif std_argument == 'metric' and args.metric == 'ridge_0_5':
                config.metric = 'ridge'
                config.params = [0.5]
            elif std_argument == 'metric' and args.metric == 'cv_ridge':
                config.metric = 'cv_ridge'
                config.params = [0.01, 0.1, 1.0, 10.0, 100]
            else:
                config.__dict__[std_argument] = args.__dict__[std_argument]
    
    print('config name: ', str(config))
    print('config: ', config.__dict__)

    #args = artificial_configs.TwoSeedsAlexNet()
    main(config)






