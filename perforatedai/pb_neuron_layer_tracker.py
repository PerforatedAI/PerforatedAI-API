'''
Functions definitions in this file do not work.
They are only here so you can see the function descriptions and
confirm they are doing what they say they are doing and 
for descriptions of inputs and outputs.
'''

class pb_neuron_layer_tracker():
    """
    Class definition for the tracker object that handles all Dendrite processing
    which is not handled internally within the modules themselves.
    """

    def initialize(doingPB: bool,
                   saveName: str,
                   maximizingScore: bool,
                   makingGraphs: bool):
        """
        Initializes variables in the tracker that the user may want different from defaults.

        Args:
            doingPB (bool): Whether or not to actually perform Perforated Backpropagation TM.
                If this is set to false Dendrites will not be added.  Can be used to test
                training pipeline without Dendrites by flipping one variable.
            saveName (str): The name to save all output networks, graphs, and csv files.
                This can be changed as a command line arg to save different runs with different
                names.
            maximizingScore (bool): When set to true Dendrites will be added when the score 
                being passed to the tracker stops going up.  True is used when maximizing 
                a validation score.  False can be used here when minimizing validation loss.
            makingGraphs (bool): Can be set to false if you don't want to save graphs as you run the system.
        """
        pass
    
    def setOptimizer(optimizer: torch.optim.optimizer):
        """
        Sets the optimizer type for the tracker to use

        Args:
            optimizer (torch.optim.optimizer): A type of optimizer for the tracker to use.
        """
        pass
    
    def setScheduler(scheduler: torch.optim.scheduler):
        """
        Sets the scheduler type for the tracker to use.

        Args:
            scheduler (torch.optim.scheduler): A type of scheduler for the tracker to use.
        """
        pass
    
    
    def setupOptimizer(model:  nn.Module, 
                        optimArgs: dict,
                        schedArgs: dict) ->
                        tuple ( torch.optim.optimizer,
                                torch.optim.scheduler)
        """
        Initializes the optimizer and scheduler using the parameters from the model.

        Args:
            model (nn.Module): The model being trained
            optimArgs (dict): Dictionary kargs to intialize the optimizer
            schedArgs (dict): Dictionary kargs to intialize the scheduler
                If this is not passed in it defaults to None and a scheduler is not created
        Returns:
            torch.optim.optimizer: initialized optimizer which was created
            torch.optim.scheduler: initialized scheduler which was created
            
            or
            
            If a scheduler is not passed in this function only returns an optimizer
            rather than a tuple
        """
        pass
    
    def setOptimizer(optimizer: torch.optim.optimizer)
        """
        If you want to handle the optimizer yourself this just gives the tracker a pointer to it
        and it will not be modified.

        Args:
            optimizer (torch.optim.optimizer): The optimizer being used for training.
        """
        pass
    
    def addExtraScore(score: float, scoreName: str)
        """
        Adds a score to the tracker for graphing and saving in addition to test and validation.
        Adding a score here will include it in csvs and graphs.

        Args:
            score (float): The score recorded at this epoch
            scoreName (str): The name to be associated with this score
        """
        pass
    
    def addTestScore(score: float, scoreName: str)
        """
        Adds a test score to the tracker for graphing and saving.
        Adding a score here will include it in csvs and graphs.
        The test score will also be used for the bestTestScores values.

        Args:
            score (float): The score recorded at this epoch
            scoreName (str): The name to be associated with this score
        """
        pass
    
    def addValidationScore(score: float, 
                            model: nn.Module,
                            saveName: str) ->
                            tuple(nn.Module,
                                  bool,
                                  bool,
                                  bool)
        """
        Adds a validation score to the tracker for graphing and saving.
        Adding a score here will include it in csvs and graphs.
        The validation score will also be used for the bestTestScores values.
        In addition to to tracking calues the validation score is what determines when to
            add Dendrites.  That means this function is what handles it.

        Args:
            score (float): The score recorded at this epoch
            model (nn.Module): The model being trained.  Should call .module() if this is
                within a DataParallel or the like.
            saveName (str): The name of the training run.  Should use the same name as in initialize()
        Returns:
            nn.Module: If new Dendrites were created or added to the network this will be a new model
                object which ahs the same values as the old one, but with the new Dendrites connected
                as required.  If the architecture did not change this just returns the original model.
            bool: the first bool returned is True if this validation score is the highest seen so far.
            bool: The second bool returned is True if the achitecture just changed.  If this value
                is True a new optimizer and scheduler must be created to point to the new model.
            bool: the final bool is True if the pbTracker determines training to be complete.  This
                will only return True when a architecture has a set of trained Dendrites connected to
                the model and the neurons have been fully trained with them to the point a switch
                is triggered.  If this architecture did not improve on the previous architecture this
                will still be False and it will try again gf.maxDendriteTries times.  If that number
                of tries has been attempted then this will be True.  When it is True that is the 
                tracker's signal that training is complete and the training cycle should be exited.
        """
        pass
    
