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
		this.maxCellStates = 2; //Set number of states in according to the model
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
		
		if (i + 1 == this.width && j + 1 == this.height) {
			se = this.grid0[0][0]; //ok
		}else if (i + 1 == this.width){
			se = this.grid0[j+1][0];
		}else if (j + 1 == this.height) {
			se = this.grid0[0][i+1];
		}else {
			se = this.grid0[j + 1][i + 1]; //ok
		}

		
		if (i - 1 < 0 && j + 1 == this.height) {
			sw = this.grid0[0][this.width - 1]; //ok
		}else if (i - 1 < 0){
			sw = this.grid0[j + 1][this.width - 1];
		}else if (j + 1 == this.height) {
			sw = this.grid0[0][i-1];
		}else {
			sw = this.grid0[j + 1][i - 1]; //ok
		}

		
		if (i + 1 == this.width && j - 1 < 0) {
			ne = this.grid0[this.height - 1][0];
		}else if (j - 1 < 0){
			ne = this.grid0[this.height - 1][i + 1];
		}else if (i + 1 == this.width) {
			ne = this.grid0[j - 1][0];
		}else {
			ne = this.grid0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = this.grid0[this.height - 1][this.width - 1];
		}else if (i - 1 < 0){
			nw = this.grid0[j - 1][this.width - 1];
		}else if (j - 1 < 0) {
			nw = this.grid0[this.height - 1][i - 1];
		}else {
			nw = this.grid0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = this.grid0[j][this.width - 1];
			e = this.grid0[j][i + 1];
		}else if (i + 1 == this.width) {
			w = this.grid0[j][i - 1];
			e = this.grid0[j][0];
		}else {
			w = this.grid0[j][i - 1];
			e = this.grid0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = this.grid0[this.height - 1][i];
			s = this.grid0[j + 1][i];
		}else if (j + 1 == this.height) {
			n = this.grid0[j - 1][i];
			s = this.grid0[0][i];
		}else {
			n = this.grid0[j - 1][i];
			s = this.grid0[j + 1][i];
		}
	}//protected void periodicBoundary(int i, int j){
	
	/**
	 * @brief reflexive boundary condition
	 *      
	 * @param i is the axis-x position of lattice
	 * @param j is the axis-y position of lattice
	 */
	protected void reflexiveBoundary(int i, int j){
		
		if (i + 1 == this.width && j + 1 == this.height) {
			se = this.grid0[j][i];  
		}else if (i + 1 == this.width){
			se = this.grid0[j][i];
		}else if (j + 1 == this.height) {
			se = this.grid0[j][i];
		}else {
			se = this.grid0[j + 1][i + 1];  
		}

		
		if (i - 1 < 0 && j + 1 == this.height) {
			sw = this.grid0[j][i];  
		}else if (i - 1 < 0){
			sw = this.grid0[j][i];
		}else if (j + 1 == this.height) {
			sw = this.grid0[j][i];
		}else {
			sw = this.grid0[j + 1][i - 1];  
		}

		
		if (i + 1 == this.width && j - 1 < 0) {
			ne = this.grid0[j][i];
		}else if (j - 1 < 0){
			ne = this.grid0[j][i];
		}else if (i + 1 == this.width) {
			ne = this.grid0[j][i];
		}else {
			ne = this.grid0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = this.grid0[j][i];
		}else if (i - 1 < 0){
			nw = this.grid0[j][i];
		}else if (j - 1 < 0) {
			nw = this.grid0[j][i];
		}else {
			nw = this.grid0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = this.grid0[j][i + 1];
			e = this.grid0[j][i + 1];
		}else if (i + 1 == this.width) {
			w = this.grid0[j][i - 1];
			e = this.grid0[j][i - 1];
		}else {
			w = this.grid0[j][i - 1];
			e = this.grid0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = this.grid0[j + 1][i];
			s = this.grid0[j + 1][i];
		}else if (j + 1 == this.height) {
			n = this.grid0[j - 1][i];
			s = this.grid0[j - 1][i];
		}else {
			n = this.grid0[j - 1][i];
			s = this.grid0[j + 1][i];
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
		
		if (i + 1 == this.width && j + 1 == this.height) {
			se = C ;//this.grid0[0][0]; //ok
		}else if (i + 1 == this.width){
			se = C; //this.grid0[j+1][0];
		}else if (j + 1 == this.height) {
			se = C; //this.grid0[0][i+1];
		}else {
			se = this.grid0[j + 1][i + 1]; //ok
		}

		
		if (i - 1 < 0 && j + 1 == this.height) {
			sw = C; //this.grid0[0][this.width - 1]; //ok
		}else if (i - 1 < 0){
			sw = C; //this.grid0[j + 1][this.width - 1];
		}else if (j + 1 == this.height) {
			sw = C; //this.grid0[0][i-1];
		}else {
			sw = this.grid0[j + 1][i - 1]; //ok
		}

		
		if (i + 1 == this.width && j - 1 < 0) {
			ne = C; //this.grid0[this.height - 1][0];
		}else if (j - 1 < 0){
			ne = C; //this.grid0[this.height - 1][i + 1];
		}else if (i + 1 == this.width) {
			ne = C; //this.grid0[j - 1][0];
		}else {
			ne = this.grid0[j - 1][i + 1];
		}
		
		if (i - 1 < 0 && j - 1 < 0) {
			nw = C; //this.grid0[this.height - 1][this.width - 1];
		}else if (i - 1 < 0){
			nw = C; //this.grid0[j - 1][this.width - 1];
		}else if (j - 1 < 0) {
			nw = C; //this.grid0[this.height - 1][i - 1];
		}else {
			nw = this.grid0[j - 1][i - 1];
		}
		
		
		if (i - 1 < 0) {
			w = C; //this.grid0[j][this.width - 1];
			e = this.grid0[j][i + 1];
		}else if (i + 1 == this.width) {
			w = this.grid0[j][i - 1];
			e = C; //this.grid0[j][0];
		}else {
			w = this.grid0[j][i - 1];
			e = this.grid0[j][i + 1];
		}	
		
		if (j - 1 < 0) {
			n = C; //this.grid0[this.height - 1][i];
			s = this.grid0[j + 1][i];
		}else if (j + 1 == this.height) {
			n = C; //this.grid0[j - 1][i];
			s = this.grid0[0][i];
		}else {
			n = this.grid0[j - 1][i];
			s = this.grid0[j + 1][i];
		}
	}//protected void constantBoundary(int i, int j, int C){
}//public class CellularAutomataModel extends SimpleCAModel 
