package models;

public class CellularAutomataModel extends SimpleCAModel {
	
	
	protected String mBoundary = "";
	
	
	protected int nw = -1, 
			       n = -1, 
			      ne = -1, 
			       w = -1, 
			       e = -1, 
			       sw = -1, 
			       s = -1, 
			       se = -1, 
			       c = -1;
	
    /*
    nw | n | ne
   ----|---|----
    w  | c |  e
   ----|---|----
    sw | s | se
     */	
	
	/**
	 * @brief constructor
	 *      
	 * @param w is number of cells the axis-x
	 * @param h is number of cells the axis-x
	 * @param boundary is one of boundary condition 
	 */
	public CellularAutomataModel(int w, int h, String boundary) {
		super(w, h);
		this.mTotalStates = 2; //Set number of states in according to the model
		this.mBoundary = new String(boundary);
	}//public CellularAutomtaModel() {
	
	
	public CellularAutomataModel() {
		super();
		
	}
	/**
	 * @brief periodic boundary condition
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 */
	protected void periodicBoundary(int i, int j){
		
		if (i + 1 == mCellX && j + 1 == mCellY) {
			se = mS0[0][0]; //ok
		}else if (i + 1 == mCellX){
			se = mS0[j+1][0];
		}else if (j + 1 == mCellY) {
			se = mS0[0][i+1];
		}else {
			se = mS0[j + 1][i + 1]; //ok
		}

		
		if (i - 1 < 0 && j + 1 == mCellY) {
			sw = mS0[0][mCellX - 1]; //ok
		}else if (i - 1 < 0){
			sw = mS0[j + 1][mCellX - 1];
		}else if (j + 1 == mCellY) {
			sw = mS0[0][i-1];
		}else {
			sw = mS0[j + 1][i - 1]; //ok
		}

		
		if (i + 1 == mCellX && j - 1 < 0) {
			ne = mS0[mCellY - 1][0];
		}else if (j - 1 < 0){
			ne = mS0[mCellY - 1][i + 1];
		}else if (i + 1 == mCellX) {
			ne = mS0[j - 1][0];
		}else {
			ne = mS0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = mS0[mCellY - 1][mCellX - 1];
		}else if (i - 1 < 0){
			nw = mS0[j - 1][mCellX - 1];
		}else if (j - 1 < 0) {
			nw = mS0[mCellY - 1][i - 1];
		}else {
			nw = mS0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = mS0[j][mCellX - 1];
			e = mS0[j][i + 1];
		}else if (i + 1 == mCellX) {
			w = mS0[j][i - 1];
			e = mS0[j][0];
		}else {
			w = mS0[j][i - 1];
			e = mS0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = mS0[mCellY - 1][i];
			s = mS0[j + 1][i];
		}else if (j + 1 == mCellY) {
			n = mS0[j - 1][i];
			s = mS0[0][i];
		}else {
			n = mS0[j - 1][i];
			s = mS0[j + 1][i];
		}
	}//protected void periodicBoundary(int i, int j){
	
	/**
	 * @brief reflexive boundary condition
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 */
	protected void reflexiveBoundary(int i, int j){
		
		if (i + 1 == mCellX && j + 1 == mCellY) {
			se = mS0[j][i];  
		}else if (i + 1 == mCellX){
			se = mS0[j][i];
		}else if (j + 1 == mCellY) {
			se = mS0[j][i];
		}else {
			se = mS0[j + 1][i + 1];  
		}

		
		if (i - 1 < 0 && j + 1 == mCellY) {
			sw = mS0[j][i];  
		}else if (i - 1 < 0){
			sw = mS0[j][i];
		}else if (j + 1 == mCellY) {
			sw = mS0[j][i];
		}else {
			sw = mS0[j + 1][i - 1];  
		}

		
		if (i + 1 == mCellX && j - 1 < 0) {
			ne = mS0[j][i];
		}else if (j - 1 < 0){
			ne = mS0[j][i];
		}else if (i + 1 == mCellX) {
			ne = mS0[j][i];
		}else {
			ne = mS0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = mS0[j][i];
		}else if (i - 1 < 0){
			nw = mS0[j][i];
		}else if (j - 1 < 0) {
			nw = mS0[j][i];
		}else {
			nw = mS0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = mS0[j][i + 1];
			e = mS0[j][i + 1];
		}else if (i + 1 == mCellX) {
			w = mS0[j][i - 1];
			e = mS0[j][i - 1];
		}else {
			w = mS0[j][i - 1];
			e = mS0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = mS0[j + 1][i];
			s = mS0[j + 1][i];
		}else if (j + 1 == mCellY) {
			n = mS0[j - 1][i];
			s = mS0[j - 1][i];
		}else {
			n = mS0[j - 1][i];
			s = mS0[j + 1][i];
		}
	}//protected void reflexiveBoundary(int i, int j){
	
	/**
	 * @brief constant boundary condition
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 * @param C is constant value, must be one of cell state
	 */
	protected void constantBoundary(int i, int j, int C){
		
		if (i + 1 == mCellX && j + 1 == mCellY) {
			se = C ;//mS0[0][0]; //ok
		}else if (i + 1 == mCellX){
			se = C; //mS0[j+1][0];
		}else if (j + 1 == mCellY) {
			se = C; //mS0[0][i+1];
		}else {
			se = mS0[j + 1][i + 1]; //ok
		}

		
		if (i - 1 < 0 && j + 1 == mCellY) {
			sw = C; //mS0[0][mCellX - 1]; //ok
		}else if (i - 1 < 0){
			sw = C; //mS0[j + 1][mCellX - 1];
		}else if (j + 1 == mCellY) {
			sw = C; //mS0[0][i-1];
		}else {
			sw = mS0[j + 1][i - 1]; //ok
		}

		
		if (i + 1 == mCellX && j - 1 < 0) {
			ne = C; //mS0[mCellY - 1][0];
		}else if (j - 1 < 0){
			ne = C; //mS0[mCellY - 1][i + 1];
		}else if (i + 1 == mCellX) {
			ne = C; //mS0[j - 1][0];
		}else {
			ne = mS0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = C; //mS0[mCellY - 1][mCellX - 1];
		}else if (i - 1 < 0){
			nw = C; //mS0[j - 1][mCellX - 1];
		}else if (j - 1 < 0) {
			nw = C; //mS0[mCellY - 1][i - 1];
		}else {
			nw = mS0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = C; //mS0[j][mCellX - 1];
			e = mS0[j][i + 1];
		}else if (i + 1 == mCellX) {
			w = mS0[j][i - 1];
			e = C; //mS0[j][0];
		}else {
			w = mS0[j][i - 1];
			e = mS0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = C; //mS0[mCellY - 1][i];
			s = mS0[j + 1][i];
		}else if (j + 1 == mCellY) {
			n = C; //mS0[j - 1][i];
			s = mS0[0][i];
		}else {
			n = mS0[j - 1][i];
			s = mS0[j + 1][i];
		}
	}//protected void constantBoundary(int i, int j, int C){
}//public class CellularAutomataModel extends SimpleCAModel 
