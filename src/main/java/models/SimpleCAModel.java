package models;

public class SimpleCAModel implements logic.CellularAutomataInterface {
	protected static final int CLASS_STATE_NOT_DEFINED                  = -1;
	protected static final int CLASS_STATE_INITIAL   					=  0;
	protected static final int CLASS_STATE_INITIAL_CONDITION 			=  1;
	protected static final int CLASS_STATE_UPDATE 						=  2;
	protected static final int CLASS_STATE_FINAL_CONDITION 				=  3;

	
	protected int 	mCellX,
				    mCellY,
				    mTimeStep,
				    mCAState,
				    mTotalStates;
	
	protected int[][] mS0, mS1;
	/**
	 * @brief default constructor
	 *      
	 */
	public SimpleCAModel() {
		this.mCellX = 0;
		this.mCellY = 0;
		this.mTimeStep = 0;
		this.mS0 = null;
		this.mS1 = null;
		this.mCAState = CLASS_STATE_NOT_DEFINED;
		this.mTotalStates = 0;
		
	}
	
	/**
	 * @brief constructor
	 *      
	 * @param w is number of cells the axis-x
	 * @param h is number of cells the axis-x
	 * @param boundary is one of boundary condition 
	 */
	public SimpleCAModel(int w, int h) {
		this.mCellX = w;
		this.mCellY = h;
		this.mTimeStep = 0;
		this.mS0 = new int[this.mCellY][this.mCellX];
		this.mS1 = new int[this.mCellY][this.mCellX];
		this.mCAState = CLASS_STATE_INITIAL;
		this.mTotalStates = 2;
	}//public CellularAutomtaModel() {
	
	@Override
	public void initialCondition() {
		// TODO Auto-generated method stub
		System.out.println("initial condition");
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		 this.mTimeStep++;
		 for (int j = 0; j < this.mCellY; j++)
			 for (int i = 0; i < this.mCellX; i++) {
				 int swp = this.mS1[j][i];
				 this.mS1[j][i] = this.mS0[j][i];
				 this.mS0[j][i] = swp;
			 }
		 	
	}

	@Override
	public void finalCondition() {
		// TODO Auto-generated method stub
		System.out.println("Final condition");
	}

	@Override
	public int getStateCA() 		{ return this.mCAState; }


	@Override
	public int getWidth()		 	{ return this.mCellX; }

	@Override
	public int getHeight() 			{ return this.mCellY;}


	@Override
	public int getStateCell(int i, int j) {
		return this.mS1[j][i];
	}

	@Override
	public void setStateCell(int i, int j, int s) {
		this.mS0[j][i] = s;
	}
	
	@Override
	public int getNumberOfStatesCell() { return this.mTotalStates; }

	@Override
	public int getTimeStep() { return this.mTimeStep; }
	
	public int getLayersSize() {return 1; }
	
	public String getLayerName(int i) {return "CURRENT STATE";} 
	
	public String getLogBasedOnLayer(int i) { return "LOG OF CURRENT STATE"; }

}
