from pathlib import Path
import os
import pandas as pd
import shutil


def lwll_to_imagenet_directory_structure(source_path, destination_path, train_labels_dict, val_labels_dict=None,
                                         sample=False):
    """
    Create an imagenet directory structure for SSL settings based on an LWLL directory structure

    train
        labelled
            <imagenet-like structure> (directories with class names containing raw image files)
        unlabelled
            UNKNOWN_class (containing raw training unlabeled images)
     val
        <imagenet-like structure> (directories with class names containing raw image files)
     test
        UNKNOWN_class (containing raw test unlabeled images)

    source_path: str or Path denoting the lwll parent dir (e.g. data/cifar100 containing cifar100_full, cifar100_sample,etc.)
    destination_path: str or Path denoting the target dir under which will be created train, test, val subdirectories
    train_labels_dict: dictionary containing {img_name: label} pairs for all training labels (under train folder)
    val_labels_dict: if specified should contain {prefix/img_name: label} pairs for validation dataset. If none, validation will be empty
    sample: if set, the sample version of the dataset will be transformed (opposed to the full version)
    nshot: integer if passed will randomly sample n-shots per class and use as labeled data and rest will be unlabelled
    seed: random seed for nshot
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    os.makedirs(os.path.join(destination_path, 'train/labelled'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'train/unlabelled/UNKNOWN_CLASS'), exist_ok=True)
    if val_labels_dict is not None:
        os.makedirs(os.path.join(destination_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'test/UNKNOWN_CLASS'), exist_ok=True)

    dataset_name = os.path.abspath(source_path).split('/')[-1]
    sfx = 'sample' if sample else 'full'

    # create train labelled
    ctr = 0
    for filename, cls in train_labels_dict.items():
        cls = str(cls)
        if not os.path.exists(destination_path/'train/labelled'/cls):
            os.mkdir(destination_path/'train/labelled'/cls)
        try:
            os.symlink(source_path/f'{dataset_name}_{sfx}/train/{filename}', destination_path/'train/labelled'/cls/filename)
            ctr +=1
        except FileExistsError:
            ctr += 1
            pass
        try:
            # remove from unlabeled in case it exists (considering an active learning situation where more labels get revealed over time)
            os.remove(destination_path/'train/unlabelled/UNKNOWN_CLASS'/filename)
        except:
            pass
    # create train unlabelled set
    unlabeled_train_files = [file for file in os.listdir(source_path/f'{dataset_name}_{sfx}/train/') if file not in train_labels_dict]
    for filename in unlabeled_train_files:
        try:
            os.symlink(source_path/f'{dataset_name}_{sfx}/train/{filename}', destination_path/'train/unlabelled/UNKNOWN_CLASS'/filename)
        except FileExistsError:
            pass
        except Exception as e:
            print(f'Error during train unlabelled creation: {e}')

    # create val - note that val dictionary must contain prefixes to define whether file is under test or train directory
    if val_labels_dict is not None:
        for filename_w_prefix, cls in val_labels_dict.items():
            cls = str(cls)
            if not os.path.exists(destination_path/'val'/cls):
                os.mkdir(destination_path/'val'/cls)
            try:
                os.symlink(source_path/f'{dataset_name}_{sfx}/{filename_w_prefix}',
                           destination_path/'val'/cls/filename_w_prefix.split('/')[-1])
            except FileExistsError:
                pass

    # create test unlabeled
    test_files = [file for file in os.listdir(source_path/f'{dataset_name}_{sfx}/test/')]
    for filename in test_files:
        try:
            os.symlink(source_path/f'{dataset_name}_{sfx}/test/{filename}', destination_path/'test/UNKNOWN_CLASS'/filename)
        except FileExistsError:
            pass
        except Exception as e:
            print(f'Error during test unlabelled creation: {e}')

    temp = set(os.listdir(os.path.join(destination_path, 'train/unlabelled/UNKNOWN_CLASS'))).\
        intersection(set(list(train_labels_dict.keys())))
    print('Intersection between train labelled and unlabelled: ', len(temp))
    print('original train size:', len(os.listdir(source_path/f'{dataset_name}_{sfx}/train/')))
    print('original test size:', len(os.listdir(source_path/f'{dataset_name}_{sfx}/test/')))
    temp = ctr + len(os.listdir(os.path.join(destination_path, 'train/unlabelled/UNKNOWN_CLASS')))
    print('Symlinks train size: ', temp)
    print('Symlinks test size: ', len(os.listdir(os.path.join(destination_path, 'test/UNKNOWN_CLASS'))))
    if val_labels_dict is not None:
        print('Validation size:', len(val_labels_dict))

def create_nshot(source_path, destination_path, train_labels_dict, val_labels_dict=None,
                                         sample=False, nshot=None, seed=123):
    if os.path.exists(destination_path):
        print('Deleting destination')
        shutil.rmtree(destination_path)

    assert nshot <= 10, 'n too high'
    df = pd.DataFrame.from_dict(train_labels_dict, orient='index').reset_index()
    df.columns = ['id', 'cls']
    new_labels = {}
    for cls in df.cls.unique():
        filtered = df[df.cls == cls]
        if filtered.shape[0] < nshot:
            print(f'{cls} has only {filtered.shape[0]} images..')
            new_labels.update({img: cls for img in filtered.id.values})
        else:
            images = filtered.sample(n=nshot, replace=False, random_state=seed)
            new_labels.update({img:cls for img in images.id.values})
    lwll_to_imagenet_directory_structure(source_path, destination_path, new_labels, val_labels_dict,
                                         sample)




if __name__ == '__main__':
    source = Path('/home/inas0003/data/external/domain_net-clipart/')
    dest = Path('/home/inas0003/data/external/domain_net-clipart_standard_10shot/')
    sfx = 'full'
    train_labels = pd.read_feather(source/f'labels_{sfx}/labels_train.feather')
    train_labels = dict(zip(train_labels.id.values, train_labels['class'].values))
    val_labels = pd.read_feather(source/f'labels_{sfx}/labels_test.feather')
    val_labels = dict(zip(val_labels.id.values, val_labels['class'].values))
    val_labels = {f'test/{k}':v for k,v in val_labels.items()}

    #lwll_to_imagenet_directory_structure(source, dest, train_labels, val_labels)
    create_nshot(source, dest, train_labels, val_labels, nshot=10, seed=000)