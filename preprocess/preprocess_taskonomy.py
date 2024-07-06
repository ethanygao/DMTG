'''Converting .png into .pkl'''
import os
from threading import Thread
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import warnings
from torchvision.transforms import v2
from torchvision import transforms
import pickle
import time
from multiprocessing import Process, Queue
LABEL_SET = ["depth_zbuffer", "normal", "segment_semantic", "keypoints2d", "edge_texture", "rgb"]
# SAVE_ROOT = os.path.expanduser("~/datasets/taskonomy_dataset/pkl_dataset")
SAVE_ROOT = os.path.expanduser("~/larger_pkl_dataset_new")
OUTPUT_SIZE = (256, 256)


class Transformer(Thread):
    def __init__(self, records, pbar, save_root, output_size):
        super().__init__()
        self.pbar = pbar
        self.records = records
        self.save_root = save_root
        if isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            self.output_size = (output_size,) * 2

    def run(self):
        def process_image(im, domain_name):
            if self.output_size is not None and self.output_size != im.size:
                im = im.resize(self.output_size, Image.BILINEAR)

            bands = im.getbands()
            if bands[0] == 'L':
                im = np.array(im)
                im.setflags(write=1)
                im = torch.from_numpy(im).unsqueeze(0)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # im = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(im)
                    # im = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])(im)
                    im = transforms.ToTensor()(im)

            mask = None
            if domain_name == 'depth_zbuffer':
                im = im.float()
                mask = im < (2 ** 13)
                mask = mask.bool()
                im -= 1500.0
                im /= 1000.0
            elif domain_name == 'edge_occlusion':
                im = im.float()
                im -= 56.0248
                im /= 239.1265
            elif domain_name == 'keypoints2d':
                im = im.float()
                im -= 50.0
                im /= 100.0
            elif domain_name == 'edge_texture':
                im = im.float()
                im -= 718.0
                im /= 1070.0
            elif domain_name == 'normal':
                im = im.float()
                im -= .5
                im *= 2.0
            elif domain_name == 'reshading':
                im = im.mean(dim=0, keepdim=True)
                im -= .4962
                im /= 0.2846
                # print('reshading',im.shape,im.max(),im.min())
            elif domain_name == 'principal_curvature':
                im = im[:2]
                im -= torch.tensor([0.5175, 0.4987]).view(2, 1, 1)
                im /= torch.tensor([0.1373, 0.0359]).view(2, 1, 1)
                # print('principal_curvature',im.shape,im.max(),im.min())

            return im, mask

        for path in self.records:
            split_parts = path.split("/")
            model = split_parts[-2]

            old_basename = os.path.basename(path)
            new_basename = f"{model}_{old_basename[:-len('_rgb.png')]}.pkl"
            new_path = os.path.join(SAVE_ROOT, new_basename)

            dirty_flag = False
            if not os.path.exists(new_path):
                res = {}
                for domain in LABEL_SET:
                    if domain == 'segment_semantic':
                        loading = path.replace('rgb', domain, 1).replace('rgb', 'segmentsemantic', 1)
                    else:
                        loading = path.replace('rgb', domain)

                    try:
                        im = Image.open(loading)
                        im, mask = process_image(im, domain)
                    except Exception as e:
                        dirty_flag = True
                        print(f"Error {e} happens when loading {loading}")
                        # print(e)
                        # break

                    res[domain] = im
                    if domain == "depth_zbuffer":
                        res["mask"] = mask

                if not dirty_flag:
                    with open(new_path, "wb") as fw:
                        pickle.dump(res, fw)
                    print(f"saving {new_path}")

            self.pbar.update(1)


class TransformerProc(Process):
    def __init__(self, records, save_root, output_size, info_q):
        super().__init__()
        self.records = records
        self.save_root = save_root
        if isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            self.output_size = (output_size,) * 2
        self.info_q = info_q

    def run(self):
        def process_image(im, domain_name):
            if self.output_size is not None and self.output_size != im.size:
                im = im.resize(self.output_size, Image.BILINEAR)

            bands = im.getbands()
            if bands[0] == 'L':
                im = np.array(im)
                im.setflags(write=1)
                im = torch.from_numpy(im).unsqueeze(0)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    im = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(im)

            mask = None
            if domain_name == 'depth_zbuffer':
                im = im.float()
                mask = im < (2 ** 13)
                mask = mask.bool()
                im -= 1500.0
                im /= 1000.0
            elif domain_name == 'edge_occlusion':
                im = im.float()
                im -= 56.0248
                im /= 239.1265
            elif domain_name == 'keypoints2d':
                im = im.float()
                im -= 50.0
                im /= 100.0
            elif domain_name == 'edge_texture':
                im = im.float()
                im -= 718.0
                im /= 1070.0
            elif domain_name == 'normal':
                im = im.float()
                im -= .5
                im *= 2.0
            elif domain_name == 'reshading':
                im = im.mean(dim=0, keepdim=True)
                im -= .4962
                im /= 0.2846
                # print('reshading',im.shape,im.max(),im.min())
            elif domain_name == 'principal_curvature':
                im = im[:2]
                im -= torch.tensor([0.5175, 0.4987]).view(2, 1, 1)
                im /= torch.tensor([0.1373, 0.0359]).view(2, 1, 1)
                # print('principal_curvature',im.shape,im.max(),im.min())

            return im, mask

        for path in self.records:
            # begin = time.time()
            split_parts = path.split("/")
            domain, model = split_parts[-4], split_parts[-2]
            im = Image.open(path)
            im, mask = process_image(im, domain)

            old_basename = os.path.basename(path)
            if domain == "segment_semantic":
                new_basename = f"{domain}_{model}_{old_basename[:-len('_domain_segmentsemantic.png')] }.pkl"
            else:
                new_basename = f"{domain}_{model}_{old_basename[:-(len(domain)+12)]}.pkl"

            pkl_path = os.path.join(self.save_root, new_basename)
            if not os.path.exists(pkl_path):
                with open(pkl_path, "wb") as fw:
                    pickle.dump(im.numpy(), fw)

            if mask is not None:
                mask_path = pkl_path.replace(domain, "mask", 1)
                if not os.path.exists(mask_path):
                    with open(mask_path, "wb") as fw:
                        pickle.dump(mask.numpy(), fw)

            self.info_q.put(1)
            # print("putting 1")
            # end = time.time()
            # print(f"running convertion time: {end-begin}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="The root dir including model sub dirs")
    parser.add_argument("--nthreads", type=int, required=True)
    parser.add_argument("--whitelist", type=str, default=None)
    args = parser.parse_args()
    args.root = os.path.expanduser(args.root)

    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    whitelist = []
    if args.whitelist is not None:
        with open(args.whitelist, "r") as fr:
            contents = fr.readlines()
            contents = [cont.strip() for cont in contents if len(cont)>1]
            whitelist.extend(contents)
        print(f"Having loading {len(whitelist)} models")

    # retrieving rgb files
    rgb_records = []
    for where, subdirs, files in tqdm(os.walk(os.path.join(args.root, "rgb"))):
    # for where, subdirs, files in tqdm(os.walk(os.path.join(args.root))):
        if len(subdirs) > 0:
            continue
        else:
            model = where.split("/")[-1]
            if len(whitelist) <= 0:
                full_path = [os.path.join(where, f) for f in files]
                rgb_records.extend(full_path)
            else:
                if model in whitelist:
                    full_path = [os.path.join(where, f) for f in files]
                    rgb_records.extend(full_path)
                else:
                    continue

    rgb_records.sort()
    records = rgb_records
    print(f"Having retrieved {len(records)} records")
    # records = []
    # for label in LABEL_SET:
    #     for rgb_path in rgb_records:
    #         if label == "segment_semantic":
    #             label_path = rgb_path.replace("rgb", label, 1).replace("rgb", "segmentsemantic", 1)
    #         else:
    #             label_path = rgb_path.replace("rgb", label)
    #         records.append(label_path)
    # records.extend(rgb_records)

    # multi-threads loading
    interval = len(records) // args.nthreads
    threads = []
    i = 1
    running_bar = tqdm(total=len(records), desc="Converting data")

    # multi-processes
    # info_q_ = Queue()
    # while i * interval < len(records):
    #     thread = TransformerProc(records[(i - 1) * interval:i * interval], SAVE_ROOT, OUTPUT_SIZE, info_q_)
    #     threads.append(thread)
    #     i += 1
    #
    # thread = TransformerProc(records[(i - 1) * interval:len(records)], SAVE_ROOT, OUTPUT_SIZE, info_q_)
    # threads.append(thread)
    #
    # # starting
    # for t in threads:
    #     t.start()
    #     # t.run()
    #
    # while True:
    #     try:
    #         info_q_.get(True, timeout=5)
    #         running_bar.update(1)
    #     except:
    #         break
    #
    # # joining
    # for t in threads:
    #     t.join()

    # mutli-threads
    while i * interval < len(records):
        thread = Transformer(records[(i - 1) * interval:i * interval], running_bar, SAVE_ROOT, OUTPUT_SIZE)
        threads.append(thread)
        i += 1

    thread = Transformer(records[(i - 1) * interval:len(records)], running_bar, SAVE_ROOT, OUTPUT_SIZE)
    threads.append(thread)

    # starting
    for t in threads:
        t.start()
        # t.run()

    # joining
    for t in threads:
        t.join()

    print("-"*5 + "Finish" + "-"*5)


