package models;

import java.io.File;

import org.python.core.PyObject;
import org.python.util.PythonInterpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

import javax.swing.JOptionPane;

public class CAModelPy extends SimpleCAModel {
	protected String mTitle = null;
	protected PythonInterpreter mPython = null;
	protected String mFileName = null;
	
	/**
	 * @brief constructor
	 *      
	 * @param w is number of cells the axis-x
	 * @param h is number of cells the axis-x
	 * @param boundary is one of boundary condition 
	 */
	public CAModelPy(String filename) {
		super();
		mFileName = new String(filename);
		//loadAndReload();
		initialCondition();
		
		
	}//public CellularAutomtaModel() {
	
	
	/**
	 * @brief load and reload script, allowing re-start all simulation
	 * 	      set F3 key to execute this method
	 */
	protected void loadAndReload() {
		mPython = null;
		mPython = new PythonInterpreter();
		mPython.execfile(mFileName);
		String name = mPython.eval("NAME").asString();
		if (name != null)
			mTitle = new String(name);
		else
			mTitle = new String("Unknow CA model ***");
		
		mCellX = 1;
		mCellY = 1;
		mTotalStates = 1;
		
		//mS0 = new int[this.mCellY][this.mCellX];
		mS1 = new int[this.mCellY][this.mCellX];
		
		/*
		mPython.exec("AC = CelularAutomata2D()");
		String name = mPython.eval("AC.getName()").asString();
		if (name != null)
			mTitle = new String(name);
		else
			mTitle = new String("Unknow CA model ***");
		
		mCellX = 1;
		mCellY = 1;
		mTotalStates = 1;
		
		mS0 = new int[this.mCellY][this.mCellX];
		mS1 = new int[this.mCellY][this.mCellX];
		mCAState = CLASS_STATE_INITIAL;
		*/
	}
	public String getTitle() { return mTitle; }
	
	@Override
	public void initialCondition() {
		// TODO Auto-generated method stub
		loadAndReload();
		
		
		mPython.exec("initial()");
		
		mCellX = mPython.eval("width").asInt();
		mCellY = mPython.eval("height").asInt();
		mTotalStates = mPython.eval("nStates").asInt();
		mS1 = new int[this.mCellY][this.mCellX];
		PyObject ret = mPython.eval("S0");
		for (int i = 0; i < mCellY; i++) {
			for (int j = 0; j < mCellX; j++) {
				int x = ret.__getitem__(i).__getitem__(j).asInt();
				mS1[i][j] = x;
				//System.out.print(Integer.toString(x) + " ");
			}//for (int j = 0; j < mCellX; j++) {
			//System.out.println("");
		}//for (int i = 0; i < mCellY; i++) {
		
		mCAState = CLASS_STATE_INITIAL;

		
	}//public void initialCondition() {

	@Override
	public void update() {
		// TODO Auto-generated method stub
		if ((mCAState != CLASS_STATE_INITIAL) && (mCAState != CLASS_STATE_UPDATE)) {
			 JOptionPane.showMessageDialog(null,
		                "A condição inicial não foi definida",
		                mTitle,
		                JOptionPane.INFORMATION_MESSAGE);
			return;
		} 
		
		mCAState = CLASS_STATE_UPDATE;
		int cont = mPython.eval("executing").asInt();
		System.out.println("Continue: " + Integer.toString(cont));
		mPython.set("ts", mTimeStep);
		mPython.set("states", mS1);
		mPython.exec("update(ts, states)");
		PyObject ret = mPython.eval("S1");
		for (int i = 0; i < mCellY; i++) {
			for (int j = 0; j < mCellX; j++) {
				int x = ret.__getitem__(i).__getitem__(j).asInt();
				mS1[i][j] = x;
				//System.out.print(Integer.toString(x) + " ");
			}//for (int j = 0; j < mCellX; j++) {
			//System.out.println("");
		}//for (int i = 0; i < mCellY; i++) {
		
		mTimeStep++;
		
		/*
		
		
		mPython.set("ts", mTimeStep);
		mPython.set("A", mS1);
		PyObject ret = mPython.eval("AC.update(ts, A)");
		for (int i = 0; i < mCellY; i++) {
			for (int j = 0; j < mCellX; j++) {
				int x = ret.__getitem__(i).__getitem__(j).asInt();
				mS1[i][j] = x;
				//System.out.print(Integer.toString(x) + " ");
			}//for (int j = 0; j < mCellX; j++) {
			//System.out.println("");
		}//for (int i = 0; i < mCellY; i++) {
		
		mTimeStep++;
		*/
	}

	@Override
	public void finalCondition() {
		// TODO Auto-generated method stub
		
	}
}//public class CAModelPy extends SimpleCAModel {
