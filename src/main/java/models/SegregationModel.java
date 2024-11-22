package models;

import java.io.File;

public class SegregationModel extends CellularAutomataModel{
	//http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/
	private final int EMPTY   = 0;
	private final int CLASS_A = 1;
	private final int CLASS_B = 2;
	private double mProbabilities[];
	private double mSimilar[]; //How similar is going to be acceptable for each group
	/*
	 * Similar is a concept where each group, in according to his probability, will 
	 * not move due to there are enough similar neighborhood. Otherwise, the group moves.
	 * It is interesting, great values for similar probability, the system never establish,
	 * otherwise the system achieves the stationary state fast. Group stops moving.
	 * Different from original idea, this model adopts different similar probability for 
	 * different class.   
	 */
	public SegregationModel(int w, int h, String boundary) {
		super(w, h, boundary);
		mProbabilities = new double[3];
		mProbabilities[EMPTY] = 0.80f;
		mProbabilities[CLASS_A] = 0.5f;
		mProbabilities[CLASS_B] = 1.0f - mProbabilities[CLASS_A];
		mSimilar = new double[2];
		
		mSimilar[CLASS_A - 1] = 0.07f;
		mSimilar[CLASS_B - 1] = 0.07f;
	}//public SegregationModel(int w, int h, String boundary) {
	
	public SegregationModel(File file) {
		super();
		/*
		 * We must initialize all parameters here, in this functions.
		 * --> see CellularAutomataModel Constructor
		 * --> see SimpleCAModel constructor
		 */
		int w = 8;
		int h = 8;
		
		
	}
	
	
	public void setProbabilities(double p[]) { mProbabilities = p; }
	
	@Override
	public void initialCondition() {
		System.out.println("initial condition - SegregationModel");
		System.out.print("Probabilities - empty: " + Double.toString(mProbabilities[0]));
		System.out.print(" - A: " + Double.toString(mProbabilities[1]));
		System.out.println(" - B: " + Double.toString(mProbabilities[2]));
		this.state = CLASS_STATE_INITIAL_CONDITION;
		
		for (int j = 0; j < this.height; j++)
			for (int i = 0; i < this.width; i++) {
				if (Math.random() <= mProbabilities[EMPTY]) {
					this.grid0[j][i] = EMPTY;
					this.grid1[j][i] = EMPTY;
					
				}else {
					if (Math.random() <= mProbabilities[CLASS_A]) {
						this.grid0[j][i] = CLASS_A;
						this.grid1[j][i] = CLASS_A;
						
					}else{
							this.grid0[j][i] = CLASS_B;
							this.grid1[j][i] = CLASS_B;	
					}
						
				}//if (Math.random() <= mProbabilities[EMPTY]) {
			}//for (int i = 0; this.width * this.height; i++) {
		
	
	}//public void initialCondition() {
	@Override
	public int getLayersSize() {return 3; }
	
	@Override
	public String getLayerName(int i) {
		if (i == 0)
			return "EMPTY SPACE";
		if (i == 1)
			return "CLASS A";
		if (i == 2)
			return "CLASS B";
		return "";
	} 
	@Override
	public void update() {
	
	}
	@Override
	public String getLogBasedOnLayer(int i) {
		return "All your bases are belong to us!\nWe get signal!!!\n";
	}
}//public class SegregationModel extends CellularAutomataModel{
