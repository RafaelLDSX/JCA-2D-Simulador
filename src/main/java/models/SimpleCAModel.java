package models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

public class SimpleCAModel implements logic.CellularAutomataInterface {
	protected static final int CLASS_STATE_NOT_DEFINED                  = -1;
	protected static final int CLASS_STATE_INITIAL   					=  0;
	protected static final int CLASS_STATE_INITIAL_CONDITION 			=  1;
	protected static final int CLASS_STATE_UPDATE 						=  2;
	protected static final int CLASS_STATE_FINAL_CONDITION 				=  3;

	
	protected int 	width,
				    height,
				    timeStep,
				    state,
				    maxCellStates;
	
	protected int[][] grid0, grid1;
	/**
	 * @brief default constructor
	 *      
	 */
	public SimpleCAModel() {
		this.width = 0;
		this.height = 0;
		this.timeStep = 0;
		this.grid0 = null;
		this.grid1 = null;
		this.state = CLASS_STATE_NOT_DEFINED;
		this.maxCellStates = 0;
		
	}
	
	/**
	 * @brief constructor
	 *      
	 * @param w is number of cells the axis-x
	 * @param h is number of cells the axis-x
	 * @param boundary is one of boundary condition 
	 */
	public SimpleCAModel(int w, int h) {
		this.width = w;
		this.height = h;
		this.timeStep = 0;
		this.grid0 = new int[this.height][this.width];
		this.grid1 = new int[this.height][this.width];
		this.state = CLASS_STATE_INITIAL;
		this.maxCellStates = 2;
	}//public CellularAutomtaModel() {
	
	@Override
	public void initialCondition() {
		// TODO Auto-generated method stub
		System.out.println("initial condition");
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		 this.timeStep++;
		 for (int j = 0; j < this.height; j++)
			 for (int i = 0; i < this.width; i++) {
				 this.grid0[j][i] = this.grid1[j][i];
			 }
		 	
	}

	@Override
	public void finalCondition() {
		// TODO Auto-generated method stub
		System.out.println("Final condition");
	}

	@Override
	public int getStateCA() 		{ return this.state; }


	@Override
	public int getWidth()		 	{ return this.width; }

	@Override
	public int getHeight() 			{ return this.height;}


	@Override
	public int getStateCell(int i, int j) {
		return this.grid1[j][i];
	}

	@Override
	public void setStateCell(int i, int j, int s) {
		this.grid0[j][i] = s;
	}
	
	@Override
	public int getNumberOfStatesCell() { return this.maxCellStates; }

	@Override
	public int getTimeStep() { return this.timeStep; }
	
	public int getLayersSize() {return 1; }
	
	public String getLayerName(int i) {return "CURRENT STATE";} 
	
	public String getLogBasedOnLayer(int i) { return "LOG OF CURRENT STATE"; }
	
	public void saveState() throws IOException{ 
		JFileChooser fileChooser = new JFileChooser();
		FileNameExtensionFilter filter = new FileNameExtensionFilter("Arquivo .dat", "dat");
    	fileChooser.setFileFilter(filter);
    	fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
    	
    	if (fileChooser.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
    		System.out.println("Arquivo selecionado.");
    		File file = fileChooser.getSelectedFile();
    		
    		//excluding extra ".dat" if it exists
    		FileWriter writer = new FileWriter(file.getName().contains(".dat") ? file.getName() : file + ".dat");
    		for(int i = 0; i < width; i++) {
    			for(int j = 0; j < height; j++) {
    				writer.write(Integer.toString(grid0[i][j]));
    			}
    		}
    		writer.flush();
    		writer.close();
    	}
		
	}
	
	public void loadState() throws IOException{
		BufferedReader reader;
		JFileChooser fileChooser = new JFileChooser();
		FileNameExtensionFilter filter = new FileNameExtensionFilter("Arquivo .dat", "dat");
    	fileChooser.setFileFilter(filter);
    	fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
    	
    	if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
    		System.out.println("File selected to load.");
    		File file = fileChooser.getSelectedFile();
    		reader = new BufferedReader(new FileReader(file));
    		char[] cellStates = new char[width * height];
    		reader.read(cellStates);
    		for(int i = 0; i < this.width; i++) {
    			for(int j = 0; j < this.height; j++) {
    				this.grid0[i][j] = Character.getNumericValue(cellStates[j + i * width]);
    				this.grid1[i][j] = Character.getNumericValue(cellStates[j + i * width]);
    			}
    		}
    		reader.close();
    	}
	}

}
