package models;

import java.util.Vector;

public class GameOfLife extends CellularAutomataModel {
	private final int DEAD = 0;
	private final int LIVE = 1;
	private int [][]mMoore = null;	

	private Vector<Integer> mLive, mDead;
	
	public GameOfLife(int w, int h, String boundary) {
		super(w, h, boundary);
		mMoore = new int[3][3];
		
		mLive = new Vector<Integer>();
		mDead = new Vector<Integer>();
	}
	

	@Override
	public void initialCondition() {
		System.out.println("initial condition");
		mCAState = CLASS_STATE_INITIAL_CONDITION;
		mLive.clear();
		mDead.clear();
		int clive = 0, cdead = 0;
		for (int j = 0; j < mCellY; j++)
			for (int i = 0; i < mCellX; i++) {
				if (Math.random() < 0.25f) {
					mS0[j][i] = LIVE;
					mS1[j][i] = LIVE;
					clive++;
				}else {
					mS0[j][i] = DEAD;
					mS1[j][i] = DEAD;
					cdead++;
				}
			}//for (int i = 0; mCellX * mCellY; i++) {
		
		mLive.add(clive);
		mDead.add(cdead);
	}//public void initialCondition() {
	
	@Override
	public void update() {
		if (mCAState == CLASS_STATE_INITIAL_CONDITION)
			mCAState = CLASS_STATE_UPDATE;
		int sum;
	    /*
        nw | n | ne
       ----|---|----
        w  | c |  e
       ----|---|----
        sw | s | se
	     */	
		int a = 0, b = 0;
		for (int j = 0; j < mCellY; j++) {
			for (int i = 0; i < mCellX; i++) {
				nw = n = ne = w = e = sw = s =  se = c = -1;
				sum = 0;
				c = mS0[j][i];
				
				if (mBoundary.compareTo("periodic") == 0) periodicBoundary(i, j);
				else if (mBoundary.compareTo("reflexive") == 0) reflexiveBoundary(i, j);
				else if (mBoundary.compareTo("constant [0]") == 0) constantBoundary(i, j, 0);
				else if (mBoundary.compareTo("constant [1]") == 0) constantBoundary(i, j, 1);

				/*
				if (nw == -1) {System.err.println("Error"); System.exit(-1);}
				if (n == -1) {System.err.println("Error"); System.exit(-1);}
				if (ne == -1) {System.err.println("Error"); System.exit(-1);}
				if (w == -1) {System.err.println("Error"); System.exit(-1);}
				if (e == -1) {System.err.println("Error"); System.exit(-1);}
				if (sw == -1) {System.err.println("Error"); System.exit(-1);}
				if (s == -1) {System.err.println("Error"); System.exit(-1);}
				if (se == -1) {System.err.println("Error"); System.exit(-1);}
				*/
				
				sum = nw + n + ne + w + e + sw + s + se;
				if ((sum == 3) && (c == 0)) {
	            	  mS1[j][i] = LIVE;
	            	  a++;
				}else if ((sum >= 2) && (sum <= 3) && (c == 1)) {
	            	  mS1[j][i] = LIVE;
	            	  a++;
				}else {
	            	  mS1[j][i] = DEAD;
	            	  b++;
				}
			
				
			}//for (int i = 0; i < mCellX; i++) {
		}//for (int j = 0; j < mCellX; j++) {
		super.update();
		mLive.add(a);
		mDead.add(b);
		
	}//public void update() {
	
	@Override
	public int getLayersSize() {return 3; }
	
	@Override
	public String getLayerName(int i) {
		return "CURRENT STATE" + Integer.toString(i+1);
	} 
	
	@Override
	public String getLogBasedOnLayer(int i) {
		String s = "";
		if (i < 1) {
			s = new String("Game of Life ");
			s += "is a Cellular Automata or Cellular Automaton - CA - which represents artificial life form. \n";
			s += "\t In this instance the lattice has size (" + Integer.toString(mCellX) + "," + Integer.toString(mCellY) + ")\n";
			s += "\t Moore is the neighborhood adopted\n";
			s += "\t Although initial state is random, the transition rule is deterministic and uses periodic boundary condition\n";
			s += "======================================================================================================================\n";
			s += "\t By Marcelo Zamith\n";
			s += "======================================================================================================================\n";
			s += "Info of " +  Integer.toString(i+1) + " layer\n";	
		}else if (i == 1) {
			s = new String("timestep;live;dead\n");
			for (int k = 0; k < mLive.size(); k++) {
				s+= Integer.toString(k) + ";" + Integer.toString(mLive.get(k)) + ";" + Integer.toString(mDead.get(k)) + "\n";
			}
		}else {
			int ones = 0;
			int zeros = 0;
			s = new String("");
			for (int j = 0; j < mCellY; j++) {
				for (int ii = 0; ii < mCellX; ii++) {
					int state = mS0[j][ii];
					if (state == 1) {
						s += "#";
						ones++;
					}
					else {
						s += "_";
						zeros++;
					}
				}//for (int i = 0; mCellX * mCellY; i++) {
				s += "\n";
			}//for (int j = 0; j < mCellY; j++) {
			s += "======================================================================================================================\n";
			s += "\t Dead cells: " + Integer.toString(zeros) + "\t" + Double.toString(((double) zeros / (double) (mCellX*mCellY)) * 100.0f) + "% \n";
			s += "\tAlive cells: " + Integer.toString(ones) + "\t" + Double.toString(((double) ones / (double) (mCellX*mCellY)) * 100.0f) + "% \n";
			
		}
		
		
		return s; 
	}

}
