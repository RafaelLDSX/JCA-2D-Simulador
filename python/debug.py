from Game_of_Life_2D import CellularAutomata2D 
if __name__ == '__main__':
    print('Debug Game of Life 2D based on python')
    CA = CellularAutomata2D()
    print(CA.getName())
    
    CA.initial()
    print("Malha:", CA.getWidth(), ", ", CA.getHeight(), "\tEstados:", CA.getNumberOfStates())
    print("Executando ? ", CA.getRunning())
    
    