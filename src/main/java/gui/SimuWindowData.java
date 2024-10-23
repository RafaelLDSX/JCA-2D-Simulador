package gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.TextArea;

import javax.swing.JInternalFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

public class SimuWindowData extends JInternalFrame{
	private JTextArea mLog = null;
	
	public void setLog(String s) { mLog.setText(new String(s));}
	public SimuWindowData(String layertitle) {
		super(layertitle, true, true, true, false);
		System.out.println("Constructor");
		mLog = new JTextArea ();
		mLog.setFont(new Font("Monospaced", Font.PLAIN, 16));
		mLog.setBackground(Color.black);
		mLog.setForeground(Color.yellow);
		mLog.setTabSize(3);
		mLog.setEditable(false);
		//ta.setLineWrap(true);
		//ta.setWrapStyleWord(false);
		//txtQuery.setBounds(10, 10, 365, 45);        
	    this.add(new JScrollPane (mLog));
		/*
		JTextArea ta = new JTextArea();
		ta.setPreferredSize(new Dimension(858,530));
		ta.setBackground(Color.GREEN);
		ta.setLineWrap(true);
		ta.setWrapStyleWord(false);
		
		this.setLayout(new FlowLayout());
		this.add(ta);
		*/
		
		this.setLocation(35, 35);
		this.setSize(800, 600);
		this.setVisible(true);
		this.show();
		
		 
		
	}
}//public class SimuWindowData extends JInternalFrame{
