#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from os.path import join 

import torch.optim as optim
from opacus.data_loader import DPDataLoader
from opacus import GradSampleModule

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from typing import Callable, List, Optional, Union
from torch.optim import Optimizer
import pickle

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"
        
#define RDP Optimizer     
class RDPOptimizer(DPOptimizer):

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        full_pmf: torch.Tensor,
        x_full_pmf: torch.Tensor,
        k: int,
        trng: torch._C.Generator,
        device
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode
            )
        self.full_pmf = full_pmf
        self.x_full_pmf = x_full_pmf
        self.k = k
        self.trng = trng
        self.device = device
        
    def add_noise(self):
        for p in self.params:
        
            _check_processed_flag(p.summed_grad)
            index_of_bins = self.full_pmf.multinomial(num_samples = p.numel(), replacement = True, \
                                                      generator = self.trng).to(device = self.device)
            bins = self.x_full_pmf[index_of_bins]
            jitter = torch.rand(p.numel(), generator=self.trng, device = self.device) #uniform from 0,1
            jitter = 1 / self.k * jitter - 1 / 2 / self.k #uniform from -1/2k to 1/2k
            noise = (bins + jitter).reshape(p.shape)
            noise *= self.max_grad_norm
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
            

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if args.verbose: print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    return 100*correct / len(train_loader.dataset)

def test(model, device, test_loader, verbose):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose: print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def rdp_make_private(model, optimizer, train_loader, sigma, clip, expected_batch_size, full_pmf, x_full_pmf, k, trng, device):

    model = GradSampleModule(model)
        
    #full_pmf, x_full_pmf, k needed to sample Cactus noise
    optimizer = RDPOptimizer(
    optimizer=optimizer,
    noise_multiplier=sigma,
    max_grad_norm=clip,
    expected_batch_size=expected_batch_size, 
    full_pmf = full_pmf,
    x_full_pmf = x_full_pmf,
    k = k,
    trng = trng,
    device = device)
    
    dp_train_loader = DPDataLoader.from_data_loader(
                      train_loader,
                      distributed=False,
                      generator = trng)
    return model, optimizer, dp_train_loader
        
def get_RDP_stuff(filename):
    with open(filename, 'rb') as f:
        loaded_parameters = pickle.load(f)
        rho = float(loaded_parameters["Our_RD"])
        noise_parameters = loaded_parameters["noise_parameters"]
    if isinstance(noise_parameters, list):
        noise_parameters = np.array(noise_parameters)
    return noise_parameters, rho
    
def get_sigma_from_cost(cost, q):
    sigma_to_power_q_times_two_to_power_q_ov_2 = cost * np.sqrt(np.pi) / gamma( (q+1)/ 2 )
    return (sigma_to_power_q_times_two_to_power_q_ov_2)**(1/q) / np.sqrt(2)
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--cost-per-dim",
        type=float,
        default=1.0,
        metavar="S",
        help="Cost per dim",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=2.0,
        metavar="q",
        help="Exponent in cost curve",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default="0",
        help="Random Seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print stuff :D ",
    )
    parser.add_argument(
        "--rdp-dir",
        type=str,
        default="",
        help="Where RDP Mechanisms are stored",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        metavar="k",
        help="1 / size of bin",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=15000,
        metavar="N",
        help="Number of bins",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.9999,
        metavar="r",
        help="Decay rate of mechanism",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2,
        metavar="beta",
        help="Order of Renyi Divergence",
    )
    
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    #set seeds 
    trng = torch.Generator(device=args.device)
    trng.manual_seed(args.seed) #set seed of generator
    torch.manual_seed(args.seed) #set seed for torch
    torch.cuda.manual_seed_all(args.seed) #set seed for GPU
    np.random.seed(args.seed) #set seed for numpy
    
    #load RDP parameters
    
    #load correct file name
    if args.q.is_integer() and args.beta.is_integer():
    	mechanism_file_name = f"noise_k{args.k}_N{args.N}_r{args.r}_cost{args.cost_per_dim}_q{int(args.q)}_beta{int(args.beta)}.pkl"
    elif args.q.is_integer() and not args.beta.is_integer():
    	mechanism_file_name = f"noise_k{args.k}_N{args.N}_r{args.r}_cost{args.cost_per_dim}_q{int(args.q)}_beta{args.beta}.pkl"
    elif not args.q.is_integer() and args.beta.is_integer():
    	mechanism_file_name = f"noise_k{args.k}_N{args.N}_r{args.r}_cost{args.cost_per_dim}_q{args.q}_beta{int(args.beta)}.pkl"
    else:
    	mechanism_file_name = f"noise_k{args.k}_N{args.N}_r{args.r}_cost{args.cost_per_dim}_q{args.q}_beta{args.beta}.pkl"
    	
    	
    if args.verbose: print(f"\n Looking for file {mechanism_file_name} in the {args.rdp_dir} directory\n")
    file_name = join(args.rdp_dir, mechanism_file_name)
    rdp_noise_params, rho = get_RDP_stuff(file_name) #grab the noise parameters and rho, i.e. Renyi Divergence
    rdp_sigma = get_sigma_from_cost(args.cost_per_dim, args.q) #return the sigma of a mean-0 Gaussian RV X s.t. E[X^q] = Cost
    
    if args.verbose: print(f"\n Effective sigma is {rdp_sigma:0.6}. This is meant for intuition, and is never used in the code.\n") 
    if args.verbose: print(f"\n Mechanism is {args.beta, rho}-RDP \n")
    full_pmf = np.hstack((rdp_noise_params[::-1][:-1], rdp_noise_params))
    x_full_pmf = np.arange(-len(rdp_noise_params) + 1, len(rdp_noise_params)) / args.k
    
    full_pmf = torch.from_numpy(full_pmf).to(device = args.device, dtype = torch.float32)
    x_full_pmf = torch.from_numpy(x_full_pmf).to(device = args.device, dtype = torch.float32)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    epoch_test_accuracy_tot = []
    if args.verbose: print(f'\n {count_parameters(SampleConvNet())} trainable parameters \n')
    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = rdp_make_private(
                model = model,
                optimizer = optimizer,
                train_loader = train_loader,
                sigma = args.cost_per_dim, #this input is never used, since x_full_pmf and full_pmf are already loaded 
                clip = args.max_per_sample_grad_norm,
                expected_batch_size = args.batch_size,
                x_full_pmf = x_full_pmf,
                full_pmf = full_pmf, 
                k = args.k,
                trng = trng,
                device = args.device
                )
            
        #print batch sizes to ensure reproducibility
        batch_sizes = []
        for xxx,yyy in train_loader:
            batch_sizes.append(len(xxx))

        if args.verbose:print("Batch sizes sampled:", batch_sizes)
        epoch_accuracy = []
        epoch_test_accuracy = []
        for epoch in range(1, args.epochs + 1):
            accuracy = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            if args.verbose: print(f'Train accuracy is {accuracy:.3} % \n')
            epoch_accuracy.append(accuracy)
            epoch_test_accuracy.append(test(model, device, test_loader, args.verbose))
        if epoch_test_accuracy_tot == []:
            epoch_test_accuracy_tot = epoch_test_accuracy
        else:
            epoch_test_accuracy_tot = np.vstack((epoch_test_accuracy_tot , epoch_test_accuracy))
        run_results.append(test(model, device, test_loader, args.verbose))


    if args.verbose:
        if args.n_runs == 1:
            print('\n RDP TEST EPOCH ACCURACY HISTORY\n ',epoch_test_accuracy)
        else:
            if not args.disable_dp:

                if args.cost_per_dim == 0:
                    print('\n NO NOISE SGD AVERAGE TEST ACCURACY \n ', repr(np.mean(epoch_test_accuracy_tot, axis = 0)))
                    print('\n NO NOISE SGD LOWEST TEST ACCURACY \n ',repr(np.min(epoch_test_accuracy_tot, axis = 0)))
                    print('\n NO NOISE SGD HIGHEST TEST ACCURACY \n ',repr(np.max(epoch_test_accuracy_tot, axis = 0)))
                else: 
                    print('\n RDP AVERAGE TEST ACCURACY \n ', repr(np.mean(epoch_test_accuracy_tot, axis = 0)))
                    print('\n RDP LOWEST TEST ACCURACY \n ',repr(np.min(epoch_test_accuracy_tot, axis = 0)))
                    print('\n RDP HIGHEST TEST ACCURACY \n ',repr(np.max(epoch_test_accuracy_tot, axis = 0)))

            else:
                print('\n VANILLA AVERAGE TEST ACCURACY \n ', repr(np.mean(epoch_test_accuracy_tot, axis = 0)))
                print('\n VANILLA LOWEST TEST ACCURACY \n ',repr(np.min(epoch_test_accuracy_tot, axis = 0)))
                print('\n VANILLA HIGHEST TEST ACCURACY \n ',repr(np.max(epoch_test_accuracy_tot, axis = 0)))

    repro_str = (
        f"RDP_mnist_{args.lr}_{args.cost_per_dim}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}_{args.n_runs}"
    )
    torch.save(epoch_test_accuracy_tot, f"{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"RDP_mnist_{repro_str}.pt")


if __name__ == "__main__":
    main()
