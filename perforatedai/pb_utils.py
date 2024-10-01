'''
Function definitions in this file do not work.
They are only here so you can see the function descriptions and
confirm they are doing what they say they are doing and 
for descriptions of inputs and outputs.
'''

def convertNetwork(model: nn.Module) -> nn.Module:
    """
    Converts a network into a network that has predefined modules converted into
    pb_neuron_layers.  Additional the global pbTracker is initialized and has all 
    the appropriate arrays and variables defined.

    Args:
        model (nn.Module): The main network that you would like to convert
            to have the ability to add Dendrite Nodes

    Returns:
        nn.Module: A new network that has the ability to have Dendrites added
    """
    pass




def loadSystem(saveName: str, saveTag: str, model: nn.Module, resetLayerVector: bool) -> nn.Module:
    """
    Loads a network and resets the global pbTracker to point to the right places and loads
    the values from the tracker when this network was saved

    Args:
        saveName (str): This should be the save name used in the trackers initialize function.
            It is used to determine what folder to load the network from.
        saveTag (str): This determines which file within that folder is loaded.
            Options include:
                'latest': The most recent save that was created
                'best_model': The model which had the highest validation score
                'beforeSwitch_X': The model right before switching modes.  Even numbers will be in normal   
                    training mode doing optimization, odd numbers will be in Dendrite training mode optimizing the dendrite inputs.  This could be used if you want to load a final set of values to determine what things looked like when a switch was triggered.
                'switch_X': The model right after switching modes.  Same as above for even and odds.
                    This could be used if you want to try a cycle again from the beginning with
                    a new set of parameters.
                'PBCount_X_startSteps_Y' These are temporary models that are saved while picking
                    an optimal learning rate to start training neurons again after adding Dendrites.
                    You will likely not ever need to load these.
        resetLayerVector (bool): A flag about whether or not to point the pbTracker to this new
            modules.  Should always be set to True for user purposes.

    Returns:
        nn.Module: The network which was saved in the associated save file.
    """
    pass

