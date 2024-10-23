# Global Constants
## The name of cellular automata
NAME = 'Game of life - Python script v 1.0'

class CellularAutomata2D:
    """! The 2D Cellular Automata class script.
    Do not change the class name
    Defines initial condition, transition rule, final condition and statistics
    """
    
    def initial(self):
        """! initial condition
        define here all setups that CA can be take
        """
        self.width   = 60
        self.height  = 30
        self.nStates = 2
        self.is_running = True


    def __init__(self):
        """! class init, do not change here
        """
        self.Name = NAME
        self.width = -1
        self.height = -1
        self.nStates = -1
        
    def getName(self):
        """! Cellular automata name 
        @return  A name of CA.
        """
        return self.Name
        
    def getWidth(self):
        """! Cellular automata space discretization in X-axis 
        do not change here
        @return  A number of cells in X-axis - mesh width
        """
        return self.width
        
    def getHeight(self):
        """! Cellular automata space discretization in Y-axis 
        do not change here
        @return  A number of cells in Y-axis - mesh height
        """
        return self.height   
    
    def getNumberOfStates(self):
        """! finite state of cells 
        do not change here
        @return  the quantity of states a cell can take
        """
        return self.nStates

    def getRunning(self):
        """! Informs if the rules is going to be applied 
        do not change here
        @return  True or False
        """
        return  self.is_running   
        
            