import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models.resnet as resnetPT
import math
import pdb
from itertools import chain
from perforatedai import globalsFile as gf




'''
LSTMCellProcessor is defined here to use as an example of how to set up processing functions.
Even though this is one class, what really happens is that the main module has one instance, 
which will use post_n1 and post_n2 and then each new Dendrite node gets a unique separate 
individual instance to use pre_d and post_d
'''
class LSTMCellProcessor():
    '''
    The neuron does eventually need to return h_t and c_t, but h_t gets modified py the Dendrite
    nodes first so it needs to be extracted in post_n1, and then gets added back in post_n2
    post_n1 is called right after the main module is called before any Dendrite processing.  
    It should return only the part of the output that you want to do Dendrite learning for.  
    '''
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do Dendrite functions
        self.c_t_n = c_t
        return h_t
    '''
    post_n2 is called right before passing final value forward, should return everything that 
    gets returned from main module
    h_t at this point has been modified with Dendrite processing
    '''
    def post_n2(self, *args, **kawrgs):
        h_t = args[0]
        return h_t, self.c_t_n
    '''
    Input to pre_d will be (input, (h_t, c_t))
    pre_d does filtering to make sure Dendrite is getting the right input.  This typically
    would be done in the training loop.  For example, with an LSTM this is where you check
    if it's the first iteration or not and either pass the Dendrite the regular args to the
    neuron or pass the Dendrite its own internal state.
    '''
    def pre_d(self, *args, **kwargs):
        h_t = args[1][0]
        #if its the initial step then just use the normal input and zeros
        if(h_t.sum() == 0):
            return args, kwargs
        #if its not the first one then return the input it got with its own h_t and c_t to replace parents
        else:
            return (args[0], (self.h_t_d, self.c_t_d)), kwargs
    '''
    For post processing post_d just gets passed the output, which is (h_t,c_t). Then 
    it wants to only pass along h_t as the output for the function to be passed to the parent
    while retaining both h_t and c_t.  post_d saves what needs to be saved for next time and
    passes forward only the Dendrite part that will be added to the parent
    '''
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t

# Similar to the above but for GRU. 
# A visual example of this one is shown in customization.md
class GRUProcessor():
    def post_n1(self, *args, **kawrgs):
        output = args[0][0]
        h_t = args[0][1]
        self.NHT = h_t
        return output
    def post_n2(self, *args, **kawrgs):
        output = args[0]
        return output, self.NHT
    def pre_d(self, *args, **kwargs):
        if(args[1].sum() == 0):
            return args, kwargs
        else:
            return (args[0], self.DHT), kwargs
    def post_d(self, *args, **kawrgs):
        output = args[0][0]
        h_t_d = args[0][1]
        self.DHT = h_t_d
        return output

# After defining a processor add it to these lists in global file to let the system know
gf.moduleNamesWithProcessing.append('GRU')
gf.modluesWithProcessing.append(nn.GRU)
gf.moduleProcessingClasses.append(GRUProcessor)

# General multi output processor for any number that ignores later ones
class multiOutputProcesser():
    def post_n1(self, *args, **kawrgs):
        out = args[0][0]
        extraOut = args[0][1:]
        self.extraOut = extraOut
        return out
    def post_n2(self, *args, **kawrgs):
        out = args[0]
        if(type(self.extraOut) == tuple):
            return (out,) + self.extraOut
        else:
            return (out,) + (self.extraOut,)
    def pre_d(self, *args, **kwargs):
        return args, kwargs
    def post_d(self, *args, **kawrgs):
        out = args[0][0]
        return out





class PBSequential(nn.Sequential):
        def __init__(self, layerArray):
            super(PBSequential, self).__init__()
            self.model = nn.Sequential(*layerArray)
        def forward(self, x):
            return self.model(x)


'''
This is an example of a custom module that may need to be done in addition to adding
blocks to the modulesToConvert.  Specifically it shows adding batch norm into a PBSequential block
PBSequential is used because normalization layers cause problems for correlation learning.
'''

class ResNetPB(nn.Module):
    def __init__(self, otherResNet):
        super(ResNetPB, self).__init__()
        
        self._norm_layer = otherResNet._norm_layer

        self.inplanes = otherResNet.inplanes
        self.dilation = otherResNet.dilation
        self.groups = otherResNet.groups
        self.base_width = otherResNet.base_width
        self.b1 = gf.PBSequential([
             otherResNet.conv1,
             otherResNet.bn1]
        )

        self.relu = otherResNet.relu
        self.maxpool = otherResNet.maxpool
        for i in range(1,5):
            setattr(self, 'layer' + str(i), self._make_layerPB(getattr(otherResNet,'layer' + str(i)),otherResNet, i))
        self.avgpool = otherResNet.avgpool
        self.fc = otherResNet.fc

    def _forward_impl(self, x):
        x = self.b1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
'''
PAIDataParallel is a class which does what is required behind the scenes to allow the pbTracker
to function properly with tensors being processed on multiple GPUs.  Current implementation
is a bit slow, Plan to get rid of this in the future.
'''
class PAIDataParallel(nn.DataParallel):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def gather():
        """
        Gathers and combines tensors on all GPUs
        """
        pass
