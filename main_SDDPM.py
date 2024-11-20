import copy
import json
import os
import warnings
import numpy as np
import wandb
# from data import ImageNet,LSUNBed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from tqdm import trange
import random

from diffusion import GaussianDiffusionTrainer,GaussianDiffusionSampler,LatentGaussianDiffusionTrainer,LatentGaussianDiffusionSampler
from model import Spk_UNet
from score.both import get_inception_and_fid_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
## argument parsing ##
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--train', action='store_true', default=False, help='train from scratch')
parser.add_argument('--eval', action='store_true', default=False, help='load ckpt.pt and evaluate FID and IS')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--sample_type', type=str, default='ddpm', help='Sample Type')
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb to log training')

parser.add_argument("--encoding", default='rate', type=str)
# Spiking UNet
parser.add_argument('--ch', default=128, type=int, help='base channel of UNet')
parser.add_argument('--ch_mult', default=[1, 2, 2, 4], help='channel multiplier')
parser.add_argument('--attn', default=[], help='add attention to these levels')
parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
parser.add_argument('--img_size', default=32, type=int, help='image size')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')
parser.add_argument('--timestep', default=4, type=int, help='snn timestep')
parser.add_argument('--img_ch', type=int, default=3, help='image channel')
# Gaussian Diffusion
parser.add_argument('--beta_1', default=1e-4, type=float, help='start beta value')
parser.add_argument('--beta_T', default=0.02, type=float, help='end beta value')
parser.add_argument('--T', default=1000, type=int, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', help='predict variable:[xprev, xstart, epsilon]')
parser.add_argument('--var_type', default='fixedlarge', help='variance type:[fixedlarge, fixedsmall]')
# Training
parser.add_argument('--resume', default=False, help="load pre-trained model")
parser.add_argument('--resume_model', type=str, help='resume model path')
parser.add_argument('--lr', default=2e-4, help='target learning rate')
parser.add_argument('--grad_clip', default=1., help="gradient norm clipping")
parser.add_argument('--total_steps', type=int, default=500000, help='total training steps')
parser.add_argument('--warmup', default=5000, help='learning rate warmup')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='workers of Dataloader')
parser.add_argument('--ema_decay', default=0.9999, help="ema decay rate")
parser.add_argument('--parallel', default=True, help='multi gpu training')
# Logging & Sampling
parser.add_argument('--logdir', default='./log', help='log directory')
parser.add_argument('--sample_size', type=int,default=64, help="sampling size of images")
parser.add_argument('--sample_step', type=int,default=5000, help='frequency of sampling')
# Evaluation
parser.add_argument('--save_step', type=int,default=0, help='frequency of saving checkpoints, 0 to disable during training')
parser.add_argument('--eval_step', type=int,default=0, help='frequency of evaluating model, 0 to disable during training')
parser.add_argument('--num_images', type=int,default=50000, help='the number of generated images for evaluation')
parser.add_argument('--fid_use_torch', default=True, help='calculate IS and FID on gpu')
parser.add_argument('--fid_cache', default='./stats/cifar10.train.npz', help='FID cache')
parser.add_argument('--num_step', type=int,default=1000, help='number of sampling steps')
parser.add_argument('--pre_trained_path', default='./pth/1224_4T.pt', help='FID cache')

args = parser.parse_args()


# device = torch.device('cuda:0')
# device = torch.device("mps")
device = torch.device('cpu')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, args.warmup) / args.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, args.img_ch, args.img_size, args.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
            grid = (make_grid(batch_images[:64,...]) + 1) / 2
            save_image(grid, 'ddpm.png')
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def set_range(X):
    return 2 * X - 1.


def train():
    if args.dataset == 'cifar10':
        dataset = CIFAR10(
            root='/home/dataset/Cifar10', train=True, download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        
    elif args.dataset == 'celeba':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.CenterCrop(148),
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.ImageFolder(root='/home/dataset/CelebA/celeba', 
                                                   transform=transform)
        
    elif args.dataset == 'fashion-mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.FashionMNIST(root='/home/dataset/FashionMnist', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
    
    elif args.dataset == 'mnist':
        SetRange = torchvision.transforms.Lambda(set_range)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.MNIST(root='dataset/Mnist', 
                                            train=True, 
                                            download=True,
                                            transform=transform)
        
    elif args.dataset == 'lsun':
        dataset = LSUNBed()
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True)
                                        
    datalooper = infiniteloop(dataloader)

    print(f'-------Starting loading {args.dataset} Dataset!-------')
    

    # model setup
    net_model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Loading Resume model from {args.resume_model}')
        net_model.load_state_dict(ckpt['net_model'], strict=True)
    else:
        print('Training from scratch')


    trainer = GaussianDiffusionTrainer(
    net_model, float(args.beta_1), float(args.beta_T), args.T).to(device)

    net_sampler = GaussianDiffusionSampler(
        net_model, float(args.beta_1), float(args.beta_T), args.T, args.img_size,
        args.mean_type, args.var_type).to(device)
    
    if args.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler).to(device)




    # log setup
    if not os.path.exists(os.path.join(args.logdir,'sample')):
        os.makedirs(os.path.join(args.logdir, 'sample'))
    x_T = torch.randn(int(args.sample_size), int(args.img_ch), int(args.img_size), int(args.img_size))
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2

    save_image(grid, os.path.join(args.logdir,'sample','groundtruth.png'))

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    # start training
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0.float()).mean()
            loss.backward()

            if args.wandb:
                wandb.log({'training loss': loss.item()})

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), args.grad_clip)
            optim.step()
            pbar.set_postfix(loss='%.3f' % loss)

            ## reset SNN neuron
            functional.reset_net(net_model)

            # sample
            # print(f'Sample at {step} step')
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = net_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        args.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    ## log to wandb 
                    if args.wandb:
                        wandb.log({'sample': [wandb.Image(grid, caption='sample')]})
                    
                net_model.train()

            # save
            # print(f'Save model at {step} step')
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                save_path = str(step) +  'ckpt.pt'
                torch.save(ckpt, os.path.join(args.logdir,save_path))

            # evaluate
            # print(f'Evaluate at {step} step')
            if args.eval_step > 0 and step % args.eval_step == 0 and step > 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                }
                pbar.write(
                    "%d/%d " % (step, args.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
               
                with open(os.path.join(args.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    


def eval():
    # model setup
    model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)
    
    ckpt_path = args.pre_trained_path 
    ckpt1 = torch.load(ckpt_path)['net_model']
    print(f'Successfully load checkpoint!')


    model.load_state_dict(ckpt1)
    model.eval()    

    sampler = GaussianDiffusionSampler(
        model, float(args.beta_1), float(args.beta_T), args.T, img_size=int(args.img_size),
        mean_type=args.mean_type, var_type=args.var_type,sample_type=args.sample_type,sample_steps=args.num_step).to(device)
    if args.parallel:
        sampler = torch.nn.DataParallel(sampler)

    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, int(args.img_ch), int(args.img_size), int(args.img_size)))
            batch_images = sampler(x_T.to(device))
            batch_images = batch_images.cpu()
            images.append((batch_images + 1) / 2)           
        images = torch.cat(images, dim=0).numpy()
    print(images.shape)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True) 
    print(f'IS: {IS}, IS_std: {IS_std}, FID: {FID}')


def main():
    if args.wandb:
        ## wandb init ##
        wandb.init(project="spike_diffusion", name=str(args.dataset)+str(args.sample_type))
        # suppress annoying inception_v3 initialization warning #
        warnings.simplefilter(action='ignore', category=FutureWarning)

    seed_everything(42)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    # app.run(main)
    main()
