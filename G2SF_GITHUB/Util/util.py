import numpy as np
import math
import random
import torch
from torchvision import transforms
from PIL import ImageFilter
from sklearn import random_projection
from tqdm import tqdm
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2np(tensor):
    return tensor.cpu().data.numpy() if tensor is not None else None


class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map


def de_normalizer(x):
    x = (((x.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN) * 255.)
    x[x >= 255] = 255
    x[x <= 0] = 0
    return x.astype(np.uint8)


def normalizer(x):
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transformation(x).numpy()


def get_coreset_idx(z_lib, n=1000, eps=0.90, float16=True, force_cpu=False, print_=True):
    if print_:
        print(f"Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps, random_state=None)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        if print_:
            print(f"DONE. Transformed dim = {z_lib.shape}.")
    except ValueError:
        print("Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()

    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    iterator = tqdm(range(n - 1)) if print_ else range(n - 1)
    for _ in iterator:
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)

        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    torch.cuda.empty_cache()
    return torch.stack(coreset_idx)


def get_coreset_idx_nn(z_lib, project_dim=256, n=1000, force_cpu=False, float16=True, print_=True):
    init_seeds(0)
    if print_:
        print(f"Linear mapping. Start dim = {z_lib.shape}.")
    mapper = torch.nn.Linear(z_lib.shape[1], project_dim, bias=False)

    with torch.no_grad():
        z_lib = mapper(z_lib).detach()

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx, device=z_lib.device)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

    if print_:
        print(f"DONE. Transformed dim = {z_lib.shape}.")

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()

    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    iterator = tqdm(range(n - 1)) if print_ else range(n - 1)
    for _ in iterator:
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)

        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))
    return torch.stack(coreset_idx)


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.scaled_lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lr_rate = param_group['lr']
    return lr_rate


def setting_lr_parameters(args):
    args.scaled_lr_decay_epochs = [i * args.meta_epochs // 100 for i in args.lr_decay_epochs]
    print('LR schedule: {}'.format(args.scaled_lr_decay_epochs))
    if args.lr_warm:
        args.lr_warmup_from = args.lr / 10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.meta_epochs)) / 2
        else:
            args.lr_warmup_to = args.lr


class MetricRecorder:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))



