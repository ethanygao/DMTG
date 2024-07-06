from yacs.config import CfgNode as CN
_C = CN(new_allowed=True)
_C.env = CN(new_allowed=True)
_C.env.gpus = [0]
_C.env.meta_arch = False
_C.env.nprocs = 0


_C.dataset = CN(new_allowed=True)
_C.dataset.type = 'train'
_C.dataset.name = 'celeba_a'
_C.dataset.tasks_name = [] # task name

# common
_C.dataset.args = CN(new_allowed=True)
_C.dataset.args.data_dir = ""
# for celeb-a
_C.dataset.args.split = 'train'

# for taskonomy
_C.dataset.args.label_set = []
_C.dataset.args.model_whitelist = ""
_C.dataset.args.model_limit = ""
_C.dataset.args.return_filename = False
_C.dataset.args.augment = False

# for training dataset
_C.train_dataset = CN(new_allowed=True)
_C.train_dataset.args = CN(new_allowed=True)

# for evaluating dataset
_C.val_dataset = CN(new_allowed=True)
_C.val_dataset.args = CN(new_allowed=True)

# for optimizer
_C.optimizer = CN(new_allowed=True)
_C.loss = CN(new_allowed=True)
_C.loss.name = 'naive'

# for scheduler
_C.scheduler = CN(new_allowed=True)
_C.scheduler.args = CN(new_allowed=True)

_C.eval = CN(new_allowed=True)
_C.eval.ref_info = ''

_C.run = CN(new_allowed=True)
_C.run.mode = 'train' # ori eval
_C.run.batch_size = 256 # 
_C.run.res_dir = ''
_C.run.load_ckpt_dir = ''
_C.run.seed = 22
_C.run.exp_name = ''
_C.run.finetune = False
_C.run.is_classify = False
_C.run.is_iou = False
_C.run.tau = 4.
_C.run.resume = ''
_C.run.pretrain = ''

# for model
_C.meta_arch = CN(new_allowed=True)
_C.meta_arch.args = CN(new_allowed=True)
# _C.meta_arch.args.backbone = CN(new_allowed=True)
_C.meta_arch.args.group_bone = CN(new_allowed=True)
_C.meta_arch.args.decoders = CN(new_allowed=True)

# _C.meta_arch.args.backbone.args = CN(new_allowed=True)
_C.meta_arch.args.group_bone.args = CN(new_allowed=True)
_C.meta_arch.args.decoders.args = CN(new_allowed=True)
