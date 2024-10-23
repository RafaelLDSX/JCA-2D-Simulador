package logic;

/**
 * @brief  interface used by SimuWindow.java 
 *         so that it can execute CA instructions
 * 
 * @class CellularAutomataInterface
 */
public interface CellularAutomataInterface {
	
	/**
	 * @brief  initial condition of simulation
	 */
	public void initialCondition();
	
	/**
	 * @brief  transition method 
	 * 
	 * */
	public void update();

	/**
	 * @brief  execute the final instruction of simulation
	 *         such as statistics and/or save files
	 * 
	 */
	public void finalCondition();
	

	
	/**
	 * @brief  informs the CA code state
	 *         
	 * @return Cellular Automata class state. It means 
	 *          which part of simulation the code is
	 */
	public int getStateCA();
	
	/**
	 * @brief  interface used by SimuWindow.java 
	 *         so that it can execute CA instructions
	 *         
	 * @return lattice width
	 */
	public int getWidth();
	
	/**
	 * @brief  interface used by SimuWindow.java 
	 *         so that it can execute CA instructions
	 *         
	 * @return lattice height
	 */
	public int getHeight();
	
	/**
	 * @brief  interface used by SimuWindow.java 
	 *         so that it can execute CA instructions
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 * @return the cell state
	 */
	public int getStateCell(int i, int j);
	
	/**
	 * @brief  interface used by SimuWindow.java 
	 *         so that it can execute CA instructions
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 * @param s is the state
	 */
	public void setStateCell(int i, int j, int s);
	
	/**
	 * @brief  interface used by SimuWindow.java 
	 *         so that it can execute CA instructions
	 * 
	 *  @return total of states of CA
	 */
	public int getNumberOfStatesCell();
	
	/**
	 * @brief  return the current time step
	 * 
	 *  @return the current time step
	 */
	public int getTimeStep();
	
}
