package gui;

import javax.swing.JInternalFrame;
import models.CellularAutomataModel;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.filechooser.FileSystemView;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.Scanner;
public class Settings extends JInternalFrame implements ActionListener{
    private JTextArea mText;
    private JFrame mFrame;
	private Document mXML = null;
	
	public Settings(String modelname) {
		super(modelname, true, true, true, false);
		mFrame = new JFrame("editor");
		mText = new JTextArea();
		
        // Create a menubar
        JMenuBar mb = new JMenuBar();
 
        // Create amenu for menu
        JMenu m1 = new JMenu("File");
 
        // Create menu items
        JMenuItem mi1 = new JMenuItem("Default");
        JMenuItem mi2 = new JMenuItem("Open");
        JMenuItem mi3 = new JMenuItem("Save");
        JMenuItem mi9 = new JMenuItem("Run");
 
        // Add action listener
        mi1.addActionListener(this);
        mi2.addActionListener(this);
        mi3.addActionListener(this);
        mi9.addActionListener(this);
 
        m1.add(mi1);
        m1.add(mi2);
        m1.add(mi3);
        m1.add(mi9);
 
        
        mb.add(m1);
        
        
        setDefault();
        mFrame.setJMenuBar(mb);
        mText.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 16));
        mText.setLineWrap(true);
        JScrollPane pane = new JScrollPane();
        pane.setViewportView(mText);
        mFrame.add(pane, BorderLayout.CENTER);
        //	mFrame.add(mText);
        mFrame.setSize(800, 600);
        mFrame.show();
	}
	
	private void setDefault() {
		String sxml = new String();
        
        sxml = "<?xml version=\"1.0\"?>\n";
        sxml += "<board>\n";
        sxml += "    <width>8</width>\n";
        sxml += "    <height>8</height>\n";
        sxml += "        <classes>\n";
        sxml += "        	<classe name='classe A'> (0,0),(1,1),(2,2) </classe>\n";
        sxml += "        	<classe name='classe B'> (2,0),(2,1),(2,3) </classe>\n";
        sxml += "        </classes>\n";
        sxml += "</board>\n";
        
        mText.setText(sxml);
	}
	
	@Override
	public void actionPerformed(ActionEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	
	
}
