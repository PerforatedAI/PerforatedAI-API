# Customization
This README describes how to customize your system to work better with Perforated Backpropagation<sup>tm</sup>.  If you are working with anything other than a simple MLP with linear and conv layers it is likely you will need some of the sections of this README to get your system running.
    
## 1 - Network Initialization

Network initialization is the only complicated part of this process that sometimes requires thought and experimentation. This section details what needs to be done and why, but check the "Changes of Note" sections of each of the examples to see descriptions of what we did and when to try to get a feel for what you should do with your network.  As a general rule though, you want to make sure everything other than nonlinearities are contained within PAI modules so that each Dendrite block performs the same processing as the associated neuron blocks.  However, complexities arise when there are multiple options to do this because of modules within modules where you can convert the whole thing, or each sub-module with the options below. 

### 1.1 - Setting Which Modules to Use for Dendrite Learning

This is often the part that has some complexity.  If your network is all simple layers with linear or conv layers and nonlinearities, they will be converted automatically.  However, most networks have more complicated learning modules.  Performance is often better when these modules are grouped as a single PAI module as opposed to PAI-ifying each module within them.  To tell the system that it must convert some blocks add them with the following option.  It can be good to do some experimentation with what level of depth you want to block things off, i.e. many smaller modules or fewer large modules. They can be added with the function below before convertNetwork is called.

    gf.moduleNamesToConvert.append('moduleName')

Along the same lines, all normalization layers should be contained in blocks.  This always improves performance so it is checked for in the initialization function.  If they are not in a module already, simply add them to a PBSequential with whatever is before them.  For example:

    gf.PBSequential([normalLayer, normalizationLayer])
    
#### 1.1.1 - How to Tell Modules Which are not Tagged
When you first call convertNetwork Perforated AI will print a list of all parameters which have not been wrapped.  It is not required that all modules are wrapped, but any that are not wrapped will not benefit from Perforated AI optimization.  It is reccomended to wrap everything, but if you are having trouble with the processing in the following section for some modules it is ok to just skip them.  The only modules that are automatically converted are nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, and PBSequential.  The list will look like this:

    The following params are not wrapped or are wrapped as part of a larger module.
    ------------------------------------------------------------------

    ...

    ------------------------------------------------------------------
    Press enter to confirm you do not want them to be refined

Once you have seen this list and are sure you want the ones that are not wrapped to not be wraped you can set it to be ignored in the future with:

    gf.missedOnesConfirmed = True
    
### 1.2 - Setting up Processing for Complex Modules
Finally, if any of the modules you are converting have a custom forward that has more than one tensor as input or output, for example a GRU taking in the previous and current states, you will need to write processing functions.  Please check out pb_models for examples of how to create a processing function for a block.  Once they are written add them with the following block of code. Make sure you do this in order.  They are just added to two arrays which assumes they are added in pairs, When writing these they also must not be local, i.e. within another class as a member function.

    gf.moduleNamesWithProcessing.append('GRU')
    # This processor lets the dendrites keep track of their own hidden state
    gf.moduleByNameProcessingClasses.append(PBM.GRUProcessor)

Other included examples include:

    gf.moduleNamesWithProcessing.append('ModuleWithMultipleOutputs')
    # This processor ignores all extra outputs after the first
    gf.moduleByNameProcessingClasses.append(PBM.multiOutputProcesser)

    
You will know this is required if you get an error similar to the following:

    AttributeError: 'tuple' object has no attribute 'requires_grad'
    
Also as a note, if you are using a GRU or LSTM in an non-traditional manner, such as passing the hidden tensor forward rather than the output, you may need to change how these processors are defined rather than using ours from pb_models. 

#### 1.2.1 - Understanding Processors
To help visualize what is happening the figure below is provided.  To think about designing a processing function, one must understand that the way Dendrites work is by outputting a single connection to each neuron.  This is implemented in PyTorch by taking the output tensor of a neuron layer, and adding the output tensor of the Dendrite layer multiplied by the corresponding weights.  This means when the Dendrite output is combined with the neuron output it must be done with a single tensor of the dimensions of the number of neurons in the layer.  This is simple if it is just a linear layer, one tensor in one tensor out, but it gets more complex when there are multiple tensors involved.

In the example below the following steps happen in the following order:
 - The input tensors are received by the PAI module.  For a GRU this will mean the input tensor and the hidden tensor, which is all zeros at the first pass.
 - The GRU Neuron receives these tensors directly and outputs the usual output of a GRU layer, a tuple of (output,hidden)
 - The first neuron postprocessor splits off the Neuron Hidden Tensor (NHT) so the single tensor output can be combined with the Dendrite's output'
 - The Dendrite Preprocessor receives these inputs but must filter them before getting to the GRU Dendrite module.  If it is the first input, it just returns them as usual.  But if it is a subsequent input where the hidden tensor is no longer all zeros it returns the Dendrite Hidden Tensor (DHT) rather than the NHT which is what would have been passed in from the training loop.
 - The GRU Dendrite receives these tensors and outputs the Dendrite (output,hidden) tuple.
 - The Dendrite Postprocessor saves the DHT to be used in future timesteps and passes forward the single tensor output that can be combined with the Neuron's output.
 - The neuron and Dendrite output's are combined.
 - The neuron second postprocessor creates a new tuple with this combined output and the NHT which was saved from postprocessor one.
 - The new tuple is returned from the PAI module which has the same format as the original module before being converted to a PAI module.
    
!["GRU Processor](processorExample.png "GRU Processor")

        
#### 2 Multiple Module Systems
Some deep learning involves components which are not single pytorch Modules.  An example might be a GAN system where the discriminator and generator are separate.  If this is the case they still must be converted together.  This can be worked around simply by creating a class such as the following:

    class Pair(nn.Module):
    def __init__(self, netG, netD):
        super(Pair, self).__init__()
        self.netG = netG
        self.netD = netD
        
Once it is created simply create one of those objects and run as follows

    model1 = create_model1()
    model2 = create_model2()
    model = Pair(model1, model2)
    model = PBU.convertNetwork(model)
    #Then set the networks directly 
    model1 = model.net1
    model2 = model.net2

Important note!  If you do the above things, make sure to also add the same steps and adjustments to the addValidationScore section.

An alternative is to call convertNetwork twice but that still needs to be tested more thoroughly.
    
### 3 - Set Abnormal Input Dimensions
Some complex networks have different input dimensions during the process.  If yours does just the setting of inputDimensions is not enough.  In these cases set inputDimensions to be the most typical case in your network.  You will then have to manually call module.setThisInputDimensions(new Indexes for Node) for any modules that stray from this. This must be called after convertNetwork.  Some examples are below.  The process is that 0 goes in the place of the nodes index, and -1s go at every other dimension.

    model.onlyRecurrentModule.setThisInputDimensions([-1,-1, 0])
    model.fullyConnectedOutputLayer.setThisInputDimensions([-1, 0])
    model.3dConvLayer.setThisInputDimensions([-1,-1,0,-1,-1])
    

This is based on the output of the layer, not the input.  Try starting without any of these and then run your network, we will tell you if there is an error and how to fix it.  If you suspect there might be more than one problem, set the following flag and they will all be printed to be able to be fixed at once.

    gf.debuggingInputDimensions = 1
    
We recommend setting this flag and if there are many problems change gf.inputDimensions in the initial settings.  Then and then do this again hopefully there will be fewer and you can do these changes with the smaller count.

### 4 Using the Pretrained Networks

If you are working with a pretrained network but you need to make some of the changes above to the architecture, what you will have to do is define a new network that takes in the initial network in the __init__ and copies all the values over.  Once you define this network you can use it by adding to the following arrays before convertNetwork:

    gf.modulesToReplace = [pretrainedModule]
    gf.replacementModules = [newPAIVersion]
    
An example of this is ResNetPB in pb_models.  Keep in mind, if you want to replace the main module of the network, just do it at the top level in the main function and do not rely on the PAI conversion portion with these two lines of code.
    
## 4 - PAIDataParallel
### 4.1 - Call to PAIDataParallel
If your network must be run on multiple GPUs call PBM.PAIDataParallel instead of torch's dataParallel before convertNetwork.  This allows the DataParallel to also keep track of Dendrite values.  However, the current implimentation does cause additional slowdown to the training process.

    gf.usingPAIDataParallel = True # This is required to eliminate the warning you get without it
    model = PBM.PAIDataParallel(model, device_ids=range(torch.cuda.device_count())).to(device)

### 4.2 - Gather Data
When using PAIDataParallel you must call gatherData directly after loss.backward()

    loss.backward()
    model.gatherData()    

## 5 Optimization

### Overfitting
Sometimes adding Dendrite nodes does just cause the system to immediately overfit.  But these can often be the best scenarios where you will be able to achieve better results with a smaller model as well.  Try reducing the width or depth of your network until you start to see a larger drop in accuracy.  Often modern architectures are designed to be extremely large because compute can be cheap and worth small accuracy increases.  This means you can often reduce the size to a fraction of the original before seeing more than a couple percentage points lost in accuracy.  Try running with a smaller model and seeing if the system still just overfits or if improvement can be found that way.

### Correlation Scores are Low
If a score is above 0.005 we typically determine that to be correlation being learned correctly.  Anything less than that is likely just random noise and something is actually going wrong.  In these cases play around with the options of 1.1 and 1.2 above.  See if other wrapping methods or other processing functions are able to achieve better correlation scores.

#### Suggestions
 - If you have a layer that gets called more than once in the forward that has been seen to cause problems

 - Sometimes models will have complicated internal mechanisms and you'll have to chunk them into additional sub-modules.  A key thing to consider when deciding if things need to be grouped is what happens after them.  If there is non module math that happens between one module and the next you might need to wrap those steps in a module.  This always includes normalization layers, but can also be things like means, applying masks, changing views.  As a rule of thumb, everything other than non-linearities should be contained within a module that is converted.

### Model doesn't Seem to Learn at All

- Make sure that optimizers are being initialized correctly.  If you are not just passing in a model in one place, make sure whatever you are doing happens at every restructuring so the new variables are being used.

- Make sure the scheduler was restarted properly.  If your learning rate is extremely low after the restructuring it may seem to not be changing.

- Make sure the optimizer, scheduler, and model are all the correct variables.  Sometimes these are updated within a function that doesn't return them or the real model is self.model but that is not overwritten by addValidationScore.

- Make sure you didn't wrap something that requires specific output values for future math down the line.  Adding the Dendrite output to these values will mess up that math.  For example, If a module ends with a Softmax layer going into NLL loss, you need to make sure the Softmax layer is not being wrapped because the output of Softmax is supposed to be probabilities, so adding them to "Dendrite probabilities" is wrong to do.  For this specific case you can also just remove it and use CrossEntropyLoss instead.

<!-- ### Things that Seem to not Work at the Moment and May be Incompatible
- Binary cross entropy loss with the output layer
- Networks which also compute features like mean and variance which are then factored into loss function
- Networks which have intermediate tensors with extremely high activations
    wavemix
- Systems which already overfit like crazy and the only way they function is by early-stopping way before training error reaches threshold
    ETSFormer-->
