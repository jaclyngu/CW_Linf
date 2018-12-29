import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
from cifar10models import *
#from helper import tanh_rescale, return_max

DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True         # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better
NUM_CLASSES = 10

def tanh_rescale(x):
    #print(x, torch.nn.Tanh(x))
    return torch.nn.Tanh()(x)

def return_max(x_tensor, y):
    if x_tensor.data > y : return x_tensor.data
    else: return y

def torch_arctanh(x):
    return 0.5*(torch.log(1+x)/(1-x)) 

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

    def _optimize(self, model, optimizer, simg, modifier, timg, tlab, const, tau):
        """

        :param model:
        :param simg:
        :param modifier: the variable to optimize over. W in paper.
        :param timg: the original image being targeted at.
        :param const: the constant we need to do binary search on. c in paper.
        :return:
        """

        #if simg is None:
        #    simg = timg.clone()
        #print(simg.size())
        newimg = (tanh_rescale(modifier + simg) / 2) #tanh_rescale do the same as tf.tanh
        #print(newimg.size())
        output = model(newimg)
        #orig_output = model(tanh_rescale(timg) / 2)

        real = torch.mean((tlab) * output)
        # TODO: why multiply by 10000 here?
        other = torch.max((1 - tlab) * output - (tlab * 10000))

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = return_max(other - real, 0.0)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = return_max(real - other, 0.0)

        z = torch.zeros(newimg.size())
        if self.cuda:
            z = z.cuda()
        loss2 = torch.sum(torch.max(torch.abs(newimg - tanh_rescale(timg) / 2) - tau, z))
        loss = const * loss1 + loss2

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        return newimg, loss


    def doit(self, model, timg, simg, lab, tau):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # convert to tanh-space
        input_var = autograd.Variable(torch_arctanh(timg * 1.999999), requires_grad=False)
        input_orig = input_var #tanh_rescale(input_var)

        start_var = autograd.Variable(torch_arctanh(simg * 1.999999), requires_grad=False)
        starts = start_var #tanh_rescale(input_var)

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(1, self.num_classes)
        if self.cuda:
            target_onehot = target_onehot.cuda()
        print(torch.unsqueeze(lab, 1))
        target_onehot.scatter_(1, torch.unsqueeze(lab, 1), 1.)
        print(target_onehot)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        # rwightman's self.init_rand skipped
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)
        optimizer = optim.Adam([modifier_var], lr=self.LEARNING_RATE)
        CONST = self.INITIAL_CONST

        while CONST < self.LARGEST_CONST:
            # try solving for each value of the constant
            print('try const', CONST)

            for step in range(self.MAX_ITERATIONS):
                s, works = self._optimize(model, optimizer, starts, modifier_var, input_orig, target_var, CONST, tau)
                #simg = s.clone()
                # it worked
                if works < .0001 * CONST and self.ABORT_EARLY:
                    get = model(s)
                    works = compare(torch.argmax(get), torch.argmax(lab))
                    if works:
                        print("Found attack!")
                        return get, model(timg), s, CONST

            CONST *= self.const_factor
        print("didn't find CONST for tau ", tau)
        return None


    def attack_single(self, model, timg, lab):
        """
        Run the attack on a single image and label
        """
        simg = timg.clone()
        tau = 1.0
        while tau > 1. / 256:
            # try to solve given this tau value
            res = self.doit(model, timg.clone(), simg.clone(), lab, tau) #why clone again?
            if res is None:
                print("Failed for the initial tau, return original img.")
                return timg

            scores, origscores, nimg, const = res
            #nimg.requires_grad = False

            if self.REDUCE_CONST: const /= 2
            # the attack succeeded, reduce tau and try again
            actualtau = torch.max(torch.abs(nimg - timg))

            if actualtau < tau:
                tau = actualtau
            print("Found attack, trying smaller Tau", tau)

            simg = nimg
            tau *= self.DECREASE_FACTOR
        return nimg

    def run(self, model, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        res = []
        for img, target in zip(imgs, targets):
            res.extend(self.attack_single(model, img.unsqueeze(0), target.unsqueeze(0)))
        return np.array(res)
















