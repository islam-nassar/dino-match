import pickle
import random

import torch
from sklearn.cluster import KMeans
from sklearn import metrics
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from eval_or_predict_knn import extract_features
from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode
from eval_knn import ReturnIndexDataset
from lwll_utils import create_nshot
import pandas as pd


def get_model_scores(model_list, source_labels_list, target_data_loader, target_labels, one_shot_data_loader, use_cuda):

    scores = []
    for model, source_labels in zip(model_list, source_labels_list):
        overlap_score = _get_class_overlap_score(source_labels, target_labels)

        target_feats = extract_features(model, target_data_loader, use_cuda)
        one_shot_feats = extract_features(model, one_shot_data_loader, use_cuda)

        target_feats = target_feats.detach().cpu()
        one_shot_feats = one_shot_feats.detach().cpu()

        ent_score = _get_entropy_score(target_feats, one_shot_feats)
        clustering_score = _get_clustering_score(target_feats.numpy(), len(target_labels))


        scores.append({'class_overlap': overlap_score,
                       'clustering_quality': clustering_score,
                       'entropy': ent_score})

    return scores


def _get_class_overlap_score(source_labels, target_labels):
    overlap = set(source_labels) & set(target_labels)
    return len(overlap) / len(target_labels)


def _get_clustering_score(feats, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(feats)
    labels = kmeans_model.labels_
    score = metrics.silhouette_score(feats, labels, metric='euclidean')

    return score


def _get_entropy_score(feats, one_shot_feats, temp=0.07):
    emb = feats.unsqueeze(dim=0).repeat(one_shot_feats.size(0), 1, 1)
    os_emb = one_shot_feats.unsqueeze(dim=1).repeat(1, feats.size(0), 1)
    sim = torch.cosine_similarity(emb, os_emb, dim=2)

    p = torch.softmax(sim/temp, dim=0)
    ent = torch.distributions.Categorical(probs=p.transpose(1, 0)).entropy()

    return ent.mean().item()


def get_model(pretrained_weights, arch='vit_small', patch_size=16):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    print(f"Model {arch} {patch_size}x{patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, pretrained_weights, 'teacher', arch, patch_size)
    model.eval()
    return model

def get_dataloaders(data_path, batch_size_per_gpu=64, num_workers=10):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = ReturnIndexDataset(os.path.join(data_path, "val" if eval else "test"), transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dataset_train = ReturnIndexDataset(os.path.join(data_path, "train/labelled"), transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return data_loader_val, data_loader_train


if __name__ == '__main__':
    lwll_dataset_path = '/home/inas0003/data/external/'

    # model_list = ['/home/inas0003/data/pretrained_models/dino_imagenet_vit_small16_pretrain.pth',
    #               '/home/inas0003/data/pretrained_models/dino_dnall_vit_s_16.pth']
    model_list = ['/home/inas0003/data/pretrained_models/dino_random_vit_s_16.pth']
    # source_class_names = list(pickle.load(Path('dino_models_classes.pkl').open('rb')).values())[:-1]
    source_class_names = [[]]
    datasets = ['domain_net-clipart', 'domain_net-sketch', 'domain_net-infograph', 'domain_net-real',
                'domain_net-quickdraw', 'domain_net-painting', 'mini_imagenet', 'mars_surface_imgs', 'msl_curiosity_imgs',
                'bu_101', 'caltech_256', 'stanford_40']

    models = [get_model(model) for model in model_list]
    scores = {}
    for dataset in datasets:
        print(f'------------------------ {dataset}----------------------------------------------')
        source = os.path.join(lwll_dataset_path, dataset)
        dest = os.path.join(lwll_dataset_path, f'{dataset}_standard')
        train_labels = pd.read_feather(Path(lwll_dataset_path) / dataset / 'labels_full/labels_train.feather')
        target_class_names = [elem.lower().replace(' ', '_') for elem in set(train_labels['class'].values)]
        train_labels = dict(zip(train_labels.id.values, train_labels['class'].values))
        val_labels = pd.read_feather(Path(lwll_dataset_path) / dataset / 'labels_full/labels_test.feather')
        val_labels = dict(zip(val_labels.id.values, val_labels['class'].values))
        val_labels = {f'test/{k}': v for k, v in val_labels.items()}
        create_nshot(source, dest, train_labels, val_labels, nshot=1, seed=random.randint(0,100))
        dataloader_val, one_shot_loader = get_dataloaders(dest, num_workers=10)
        scores[dataset] = get_model_scores(models, source_class_names, dataloader_val, target_class_names, one_shot_loader, True)
        print(f"Scores for {dataset}: {scores[dataset]}")

    pickle.dump(scores, Path('model_selection_scores_random.pkl').open('wb'))