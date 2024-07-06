import os, sys
import numpy as np
import logging
import json, pickle
import torch
from .config import _C as cfg
import random
LOGER = None

def setup_cfg(args):
    with open(f'{args.cfg}', "r") as f:
        default_cfg = cfg.load_cfg(f)
    if '_BASE_' in default_cfg:
        dir_root = os.path.dirname(args.cfg)
        base_cfg = default_cfg.pop('_BASE_')
        base_cfg_path = os.path.join(dir_root, base_cfg)
        cfg.merge_from_file(base_cfg_path)
    cfg.merge_from_other_cfg(default_cfg)
    # cfg.merge_from_file()
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def init_loger(exp_dir, log_name, comments='train'):
    loger = logging.getLogger('')
    loger.setLevel(logging.INFO)
    val_log_name = f'{log_name}.log'
    if log_name.endswith('log'):
        val_log_name = log_name
    print(f'>> Init_logger to {exp_dir}/{val_log_name}')
    fh = logging.FileHandler(os.path.join(exp_dir, val_log_name), 'a')
    fh.setLevel(logging.INFO)
    format = logging.Formatter('%(asctime)s %(message)s',\
                            datefmt='%Y/%m/%d %H:%M:%S')
    fh.setFormatter(format)
    loger.addHandler(fh)

    global LOGER
    LOGER = loger

    return loger

def print_loger(_info, loger=None):
    print(_info)
    if loger is not None:
        loger.info(_info)
    else:
        try:
            LOGER.info(_info)
        except AttributeError:
            print('>> loger not inited...')


def mkdir(path):
    '''
    description: make folder  
    param {path}
    '''
    if not isinstance(path, list):
        path = [path]
    for _path in path:
        if not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
            print(f'>> {_path} not exits, create folder...')

def load_data(fdir):
    assert os.path.exists(fdir), f'not exist: {fdir}...'
    if any(fdir.endswith(s) for s in ['pkl', 'pickles', 'pickle']):
        with open(fdir, 'rb') as fb:
            ans = pickle.load(fb)
    elif fdir.endswith('json'):
        with open(fdir, 'r') as f:
            ans = json.load(f)
    elif fdir.endswith('txt'):
        with open(fdir, 'r') as f:
            ans = f.readlines()
    elif fdir.endswith('bvh'):
        with open(fdir, 'r') as f:
            ans = f.readlines()
            ans = [e.strip() for e in ans]
    elif fdir.endswith('mat'):
        with open(fdir, 'rb+') as f:
            ans = sio.loadmat(f)
    else:
        raise NotImplementedError
    return ans
