############### PAI configuration file ###############
import torch 
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#A global pointer to the tracking class
pbTracker = []

# List of nn.Module's which should be converted to Dendrite blocks.
# Anything in this list will receive a duplicate which learns with Perforated Backpropagation
modulesToConvert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
# Same as above but add them by name such as "Conv1d" in this list
moduleNamesToConvert = []
'''
Replacement modules happen before the conversion, so replaced modules will then also be run 
through the conversion steps. This can be used if you have a pretrained model but need to
change the way some of the modules are set up.  Fill these in with a pointer to the module 
and the replacement class which takes the initial module as a parameter.  See customization
section 4 for more information.
'''
modulesToReplace = []
replacementModules = []

#If your modules require processing functions add them to this list
modluesWithProcessing = []
#Then add the processing classes to this list
moduleProcessingClasses = []
#Same as above put can pass the modules in by name
moduleNamesWithProcessing = []
moduleByNameProcessingClasses = []

'''
inputDimensions needs to be set every time. It is set to what format of tensors you are
expecting.  Node index should be set to 0, other indexes should be set to -1.  For
example, if your format is [batchsize, nodes, x, y] inputDimensions is [-1,0,-1-1].
if your format is, [batchsize, timestep, nodes] indexesBeforeNode is [-1,-1,0]
'''
inputDimensions = []

#Constants
#Percentage Improvement increase needed to call a new best validation score
improvementThreshold = 0.0001 
#Raw increase needed, if its lower than this its not really learning anyway
improvementThresholdRaw = 1e-8
#Improvement increase needed to call a new best Perforated Backpropagation score
pbImprovementThreshold = 0.01 
#Raw increase needed, if its lower than this its not really learning 
pbImprovementThresholdRaw = 1e-5

# Switch Mode settings
switchMode = -1
# Switch after every epoch
doingSwitchEveryTime = 0

# Switch after no improvement over a set number of epochs
doingHistory = 1
# Make sure these numbers are higher than the scheduler patience
nEpochsToSwitch = 10  # Number of normal epochs to cause a switch
pEpochsToSwitch = 10  # Number of Perforated Backpropagation epochs to cause a switch
capAtN = False #Makes sure PB rounds last max as long as normal rounds
# Number of epochs to average validation scores over
# Set to 1 if you don't want to use averaging
historyLookback = 5
# Initially after switches number of epochs to wait to make sure you have a fair
# initialization score before tracking any new maxes and and allowing switches to happen
# Set to 1 if you do not want to do averaging.
initialHistoryAfterSwitches = 3

# Switch after a fixed number of epochs
doingFixedSwitch = 2
fixedSwitchNum = 250
#You can set the first switch to be longer than the others for a slower initialization.
#It will not go shorter, so set this lower than fixedSwitchNum to ignore
firstFixedSwitchNum = 249 

#This is for if you set doingPB to be false and just want to run without PB learning but 
#generate the same graphs and csvs
doingNoSwitch = 3

# Typically PB nodes will be deleted if the normal learning doesn't improve after adding them.
# This will retain them anyway, generally only used when testing your GPU capacity and running
# A bunch of PB cycles in a row
retainAllPB = False

# This will test various learning rates after each PB cycle.  Often a lower initial rate is
# better so the learning doesn't jump away far from the local minimum the Dendrite nodes trained on
findBestLR = True

# Set to 1 if you want to quit as soon as one Dendrite fails
# Higher numbers will try a new set of randomly initialized Dendrites
maxDendriteTries = 5

# This number is to check how many batches to average out the initial correlation score over
initialCorrelationBatches = 100 #this should be at least 100 and up to 10% of a whole epoch


paramValsSetting = paramValsByUpdateEpoch

'''
A custom PAI module which can be used to group layers into a single block

This takes in an array of layers. For example:

    gf.PBSequential([nn.Linear(2 * hidden_dim, seqWidth),
            nn.LayerNorm(seqWidth)])
    
    This should be used for all normalization layers.
    You will get warnings if normalization layers are unwrapped.
    
'''
class PBSequential(nn.Sequential):
        def __init__(self, layerArray):
            super(PBSequential, self).__init__()
            self.model = nn.Sequential(*layerArray)
        def forward(self, x):
            return self.model(x)
