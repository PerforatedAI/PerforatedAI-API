# Debugging

## Errors where warnings are printed

    "The following layer has not properly set thisInputDimensions"


Check out the suggestions that are printed.

    Didn't get any non zero scores or a score is nan or inf.
    
This means that the Dendrites learned a correlation that was either nan or infinite.  We have only seen this happen with training pipelines where the neurons are also learning weights that are getting close to triggering an inf overflow error themselves.  See if you can add normalization layers to keep your weights within more usual ranges.

    An entire layer got exactly 0 Correlation
    
Same as above but for zero.

    Trying to call backwards but module X wasn't PAIified
    
This means something went wrong with the conversion.  The module is getting triggered for PAI modifications but also wasn't converted in a way that allowed it to initialize properly.  Look into how you set up that layer.

    Need exactly one 0 in the input dimensions
    
If this is printed it means you did not set up the input dimensions for the layer properly.  Make sure it is set up with only -1s and 0s where the 0 is the index of the neurons.

    Seeing multiple GPUs but not using PAIDataParallel.  Please either perform the PAIDataParallel
    steps from the README or include CUDA_VISIBLE_DEVICES=0 in your call

When running Perforated AI with multiple GPUs DataParallel does not automatically process things correctly.  A new DataParallel must be used which is described in the customization README section 4.

    PAIDataParallel did not call gather.
    
Look at section 4 from customization.  The model must call gatherData manually after backward.

## Input Dimentions

    'pbValueTracker' object has no attribute 'out_channels'

Look at section 3 from customization.  This explains how to set input dimensions.

## Broadcast Shape

    Values[0].normalPassAverageD += (val.sum(mathTuple) * 0.01) / fullMult
    RuntimeError: output with shape [X] doesn't match the broadcast shape [Y]
    
If you get this error it means input dimensions were not properly set.  Run again with pdb and when this error comes up print Values[0].layerName to see which layer the problem is with.  You can also print the shape of val to see what the dimensions are supposed to be.  This should be caught automatically so In our experience when this happens it means you have a layer that can accept tensors which have different dimensionality without having problems.  This is not accounted for with our software currently so just wrap that layer in a module as required so you don't need to do that. 

## Errors in forward
These usually mean the processors were not set up correctly. Look at 1.2 from customization.

    AttributeError: 'tuple' object has no attribute 'requires_grad'

This specifically is saying that you are returning a tuple of tensors rather than a single tensor.  Your processor needs to tell you how to handle this so the Dendrite only collaborates on one tensor with the neuron.

Make sure that you put the gf.modulesWithProcessing setup before the call to convertNetwork
    
## Errors in filterForward
These also usually mean the processors were not set up correctly. Look at 1.2 from customization.

    AttributeError: 'NoneType' object has no attribute 'detach'

This means you are not using the tensor that is being passed to the dendrites.  For example if you are using the default LSTM processor but using hidden_state rather than output from "output, (hidden_state, cell_state)"

## dype Errors
    
    Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

If you are not working with float data change gf.dType to whatever you are using. eg:

    gf.dType = torch.double

## Device Errors

    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

Similar as above there is a setting that defaults to using what is available.  If cuda is available but you still don't want to use it call:

    gf.device = 'cpu'
    
## Attribute Error:
    
    AttributeError: 'pb_neuron_layer' object has no attribute 'SOMEVARIABLE'

This is the error you will get if you need access an attrubute of a module that is now wrapped as a pb_neuron_layer.  All you have to do in this case is the following change.

    #model.yourModule.SOMEVARIABLE
    model.yourModule.mainModule.SOMEVARIABLE
    
## Initialize Error

    gf.pbTracker.initialize ...
    AttributeError: 'list' object has no attribute 'initialize'

This means the pbTracker was not initialized.  This generally only happens if zero layers are converted.  Make sure that at least one layer has been converted correctly.

## Saving Error

    RuntimeError: Serialization of parametrized modules is only supported through state_dict()

If this is happening it means you have a parameterized module.  You can track down what it is by running in pdb and then calling torch.save on each part of your model recursively until you get to the smallest module which flags the error.  Whatever that one is will have to be changed so you can call torch.save, or work with PAI to update things to use a state_dict.  We have seen this happen before in a model that used to work because of an updated version of pytorch.  There is not currently a workaround so if you can not convert your module to one that is able to be saved with torch.save the systems may be incompatible or you can try an earlier version of torch (pre 2.0 at least).

## setupOptimizer Error
    
    TypeError: 'list' object is not callable
    
If you get this error in setupOptimizer it means you called setupOptimizer but you did not call setOptimizer.  Be sure to call that first.


## Size Mismatch

    File "perforatedai/pb_layer.py", line X, in perforatedai.pb_layer.pb_neuron_layer.forward
    RuntimeError: The size of tensor a (X) must match the size of tensor b (X) at non-singleton dimension 
    
If you get this error it means your neurons are not correctly matched in setInputDimensions.  If your 0 is in the wrong index the tensors that get used for tracking the Dendrite to Neuron weights will be the wrong size.

## Initialization Errors

    File "perforatedai/pb_layer.py" ... perforatedai.pb_layer.pb_neuron_layer.__init__
    IndexError: list index out of range
    
This means you did something wrong with the processing classes.  We have seen this before when moduleNamesWithProcessing and moduleByNameProcessingClasses don't line up.  Remember they need to be added in order in both arrays, and if the module is "by name" the processor also has to be added to the "by name" array.



## Things not Getting Converted
The conversion script runs by going through all member variables and determining all member variables that inherit from nn.Module.  If you have any lists or non nn.Module variables that then have nn.Modules in them it will miss them.  If you have a list just put that list into a nn.ModuleList and it will then find everything.  If you do this, make sure you replace the original variable name because that is what will be used. If you use the "add_module" function this is a sign you might cause this sort of problem.  Do not currently have a workaround for non-module objects that contain module objects, just let us know if that is a situation you are in and there is a reason the top object can't also be a module.


## Different Devices

- If you are getting different device problems check if you have a Parameter being set to a device inside the init function.  This seems to cause a problem with calling to() on the main model.

- A second thing to check is if you are calling to() on a variable inside of the forward() function.  Don't do this, just put it on the right device before passing it in.

## Memory Leak

A memory leak is happening if you run out of memory in the middle of a training epoch, i.e. it had enough memory for the first batch but a later batch crashes with OOM error.  These are always a pain to debug but here's some we have caught.

    -Check if one of your layers is not being cleared during backwards.  This can build up
    if you are forwarding a module but not calling backwards even though this won't cause a
    leak without PAI in the same model.  We have seen a handful of models which calculate
    values but then never actually use them for anything that goes towards calculating loss,
    so make sure to avoid that.  To check for this you can use:
        gf.debuggingMemoryLeak = True
    - If this is happening in the validation/test loop make sure you are in eval() mode which
    does not have a backwards pass.
    - Check for your training loop if there are any tensors being tracked during the loop which
    would not be cleared every time.  One we have seen often is a cumulative loss being tracked.
    Without PAI this gets cleared appropriately, but with PAI it does not.  This can be fixed
    by adding detach() before the loss is added to the cumulative variable.
    
If these don't quickly solve it, the best thing to do would be just move on to another test.  The best method we have to track it down is to try to remove various components from your model until you can identify which one stops the leak when it is gone.

<!-- There is another method which involves tracking exactly where cuda tensors are being allocated but it's extremely difficult to track this down with it. -->

## Memory Issues Inside Docker but not Outside
If you are running with a docker container and you were not using docker before it is likely an issue with the shared memory inside the docker container.  Just run with the additional flag of shm-size like so:

    docker run --gpus all -i --shm-size=256m -v .:/pai -w /pai -t pai /bin/bash


## Optimizer Initialization Error

    -optimizer = self.memberVars['optimizer'](**optArgs)
    -TypeError: __init__() got an unexpected keyword argument 'momentum'

This can happen if you are using more than one optimizer in your program.  If you are, be sure to call gf.pbTracker.setOptimizer() again when you switch to the second optimizer and also call it as the first line in the if(restructured) block for adding validation scores.


## Debugging Docker Installation

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    >>> solved with:
    sudo apt-get install libgl1-mesa-glx

## Saving PAI 
PAI saves the entire system rather than just your model.  If you run into issues with saving such as 

    `Can't pickle local object 'train.<locals>.tmp_func'`

This likely means the optimizer or scheduler are using lambda functions.  just replace the lambda function with a defined function eg: 

    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  
    #converted to a global function
    def tmp_func(x):
        return (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    lf = tmp_func #where it was originally defined

## Extra Debugging

If you are unable to debug things feel free to contact us!  We will be happy to help you work through issues and get you running with Perforated AI.
