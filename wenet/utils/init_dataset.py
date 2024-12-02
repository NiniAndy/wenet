import copy
from typing import Optional

from wenet.dataset.asr_dataset import ASRDataset
from wenet.dataset.asr2_dataset import ASR2Dataset
from wenet.text.base_tokenizer import BaseTokenizer



def init_dataset(
        split='train',
        partition=True,
        tokenizer_dict: dict={},
        dataset='asr',
        data_type="raw",
        train_data=None,
        cv_data=None,
        dataset_conf: dict = None,
        **kwargs):

    assert dataset in ['asr', 'ssl', 'asr2']

    tokenizer = tokenizer_dict.get('tokenizer', None)
    pny_tokenizer = tokenizer_dict.get('pny_tokenizer', None)

    data_list_file = train_data if split == 'train' else cv_data

    if split != 'train':
        cv_conf = copy.deepcopy(dataset_conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        dataset_conf = cv_conf

    if dataset == 'asr':
        return ASRDataset(data_type, data_list_file, tokenizer, dataset_conf, partition)
    elif dataset == 'asr2':
        return ASR2Dataset(data_type, data_list_file, tokenizer, pny_tokenizer, dataset_conf, partition)
    else:
        from wenet.ssl.init_dataset import init_dataset as init_ssl_dataset
        return init_ssl_dataset(data_type, data_list_file, dataset_conf, partition)
