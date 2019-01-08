import torch
import numpy as np
from torch import optim
from torch import autograd
import torchvision
from torchvision import transforms
from models import *
import matplotlib.pyplot as plt
#from helper import tanh_rescale, return_max

DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True        # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better
NUM_CLASSES = 10

# cifar only, -1..1 image range
def tanh(x):
    #print(x, torch.nn.Tanh(x))
    return torch.nn.Tanh()(x)

def return_max(x_tensor, y):
    if x_tensor.data > y : return x_tensor.data
    else: return y

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


class CW_Linf:
    def __init__(self,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 reduce_const = REDUCE_CONST, decrease_factor = DECREASE_FACTOR,
                 const_factor = CONST_FACTOR, num_classes = NUM_CLASSES):
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.num_classes = num_classes
        self.cuda = True

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

    def gradient_descent(self, model):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
    
        # TODO replace hardcode
        shape = (1, 3, 32, 32)

        def doit(oimgs, labs, starts_temp, tt, CONST):
#             print(f'BEFORE arctanh, max new image: {oimgs.max()}')
#             print(f'min new image: {oimgs.min()}')
#             print(f'BEFORE arctanh, max new image: {starts_temp.max()}')
#             print(f'min new image: {starts_temp.min()}')

#             import pdb; pdb.set_trace()
            
            # convert to tanh space
            imgs = torch_arctanh(oimgs * 1.999999)
            starts = torch_arctanh(starts_temp * 1.999999)
#             print(f'AFTER arctanh, max new image: {oimgs.max()}')
#             print(f'min new image: {oimgs.min()}')
#             print(f'AFTER arctanh, max new image: {starts.max()}')
#             print(f'min new image: {starts.min()}')

            # initial tau
            tau = tt
            timg = imgs 
            tlab = labs
            const = CONST

            # changing tlab to one hot
            target_onehot = torch.zeros((1, self.num_classes), requires_grad=True, device="cuda")
            print(torch.unsqueeze(tlab, 1))
            tlab = target_onehot.scatter(1, tlab.unsqueeze(1), 1.)
            print(tlab)
            # iterate through constants, try to get highest one
            while CONST < self.LARGEST_CONST:
                
                # try solving for each value of the constant
                print('try const', CONST)
                
#                # changing tlab to one hot
#                target_onehot = torch.zeros((1, self.num_classes), requires_grad=True, device="cuda")
#                print(torch.unsqueeze(tlab, 1))
#                tlab = target_onehot.scatter(1, tlab.unsqueeze(1), 1.)
#                print(tlab)
                
                # setup the modifier variable, this is the variable we are optimizing over
                modifier = torch.zeros(shape, requires_grad=True, device="cuda")
                optimizer = optim.Adam([modifier], lr=self.LEARNING_RATE)

#                print('got past modifier!')

                # starting point for simg
                simg = starts.clone()

                for step in range(self.MAX_ITERATIONS):
                    newimg = tanh(modifier + simg) / 2
#                    print('got past newimg!')

                    output = model(newimg)
                    orig_output = model(tanh(timg)/2) # currently assumes -0.5..0.5
#                    print('got past outputs!')
                    
                    real = torch.mean((tlab) * output)
                    other = torch.max((1 - tlab) * output - (tlab * 10000))
#                    print('got past real/other!')

                    if self.TARGETED:
                        # if targetted, optimize for making the other class most likely
                        loss1 = torch.max(other - real, torch.zeros_like(real))
                    else:
                        # if untargeted, optimize for making this class least likely.
                        loss1 = torch.max(real - other, torch.zeros_like(real))
#                    print('got past loss1!')

                    loss2 = torch.sum(torch.max(torch.abs(newimg - tanh(timg) / 2) - tau, torch.zeros_like(newimg)))
                    loss = const * loss1 + loss2
#                    print('got past loss2 and loss!')

                    # old code
#                    works = loss.clone()
#                    scores = output.clone()
                    
                    optimizer.zero_grad()
                    
#                    print('got past optimizer zero grad!')
                    loss.backward(retain_graph=True)
                    
#                    print('got past loss backward!')

#                     import pdb; pdb.set_trace()
                    
                    optimizer.step()
#                    print('got past optimizer step!')

                    # it worked
                    if loss < .0001 * CONST and self.ABORT_EARLY:
                        #get = output
                        works = compare(torch.argmax(output), torch.argmax(tlab))
                        if works:
                            print("Found attack!")
                            return output, orig_output, newimg, CONST
                        
                C = CONST * self.const_factor
                CONST = C

        return doit

    def attack(self, model, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for img, target in zip(imgs, targets):
            r.extend(self.attack_single(model, img.unsqueeze(0), target.unsqueeze(0)))
        return np.array(r)

    def attack_single(self, model, img, target):
        """
        Run the attack on a single image and label
        """
        prev = img.clone()
        tau = 1.0
        const = self.INITIAL_CONST

        while tau > 1. / 256:
            # try to solve given this tau value
            res = self.gradient_descent(model)(img.clone(), target, prev.clone(), tau, const)
            if res is None:
                return prev

            scores, origscores, nimg, const = res
#             print(f'nimg max: {nimg.max()}')
#             print(f'nimg min: {nimg.min()}')

#             print(f'img max: {img.max()}')
#             print(f'img min: {img.min()}')
            
            if self.REDUCE_CONST: const =torch.div( CONST, 2)

            # the attack succeeded, reduce tau and try again
            actualtau = torch.max(torch.abs(nimg - img))

            if actualtau < tau:
                tau = actualtau

            prev = nimg.clone()
            tau = torch.mul(tau, self.DECREASE_FACTOR)

            print("Found attack, trying smaller Tau", tau)
        print("Loop is over. Returning last successful adversarial example.")
        return prev

if __name__ == "__main__":
    net = ResNet18().cuda()
    net.load_state_dict(torch.load('/home/isk22/resnet18_epoch_347_acc_94.77.pth', map_location=lambda storage, loc: storage))
    net.eval()

    def net_hacked(x):
        return net(x * 2)

    attack = CW_Linf()

    batch_size=128
    epochs=300
    means = (0.5, 0.5, 0.5)
    stddevs = (1, 1, 1) #(0.5, 0.5, 0.5)

# Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stddevs),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stddevs),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/share/cuvl/pytorch_data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='/share/cuvl/pytorch_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ims, lbls = iter(trainloader).next()
    ims, lbls = ims.cuda(), lbls.cuda()
    res = attack.attack(net_hacked, ims[0:1], torch.tensor([0]).cuda()) #lbls[0:1])V

    pert = res[0]
    display_pert = pert.detach().cpu().numpy().transpose(1,2,0)
    print('original label ', lbls[0:1], '\n')
    print('original pred', net_hacked(ims[0:1]), '\n')
    print('ad ex pred', net_hacked(pert.unsqueeze(0)), '\n')
    print('original max min', ims[0:1].max(), ims[0:1].min(), '\n')
    print('ad ex max min', pert.max(), pert.min(), '\n')
