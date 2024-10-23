/**
 * 
 */
package gui;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JDesktopPane;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.text.NumberFormatter;

import models.CAModelPy;
import models.SegregationModel;
/**
 * 
 */
public class MainWindow  extends JFrame {
		private static final int BOUNDARY = 50;
		private static final int ITENS  = 9;
		
		 
		private static final int EXIT_CLOSE				= 0;
		private static final int LOAD 					= 1;
	 
		private static final int ABOUT          		= 2;
		public static final int GAME_OF_LIFE   			= 3;
		public static final int PRISONERS_DILEMMA  		= 4;
		public static final int PREDATOR_PREY     		= 5;
		public static final int LOAD_MODEL_PY      		= 6;
		public static final int NATIVE_SIMULATION       = 7;
		public static final int SEGREGATION_2           = 8;
		
	    private JPanel   mPanel   = null;
	    private JMenuBar mMenuBar = null;
	    private JMenu    mMenu    = null,
	                     mSubmenu = null;
	    private JMenuItem mMenuItem = null;
	    
	    private JDesktopPane mPainel =  null;
	    private int []mMenuHash;
	    
    	public MainWindow() {
            this.setTitle("Cellular automata simulation");
            //this.setExtendedState(JFrame.MAXIMIZED_BOTH); // To maximize a frame
            //int bordas = 50;
            
            Dimension tela = Toolkit.getDefaultToolkit().getScreenSize();
            setBounds(BOUNDARY, BOUNDARY, tela.width  - BOUNDARY * 2 , tela.height - BOUNDARY * 2);
            
            mPainel = new JDesktopPane(); 
            mPainel.setBackground(Color.LIGHT_GRAY);
        	

            
            this.setContentPane(mPainel);
            
            this.createMenu();
       
            //this.setLayout(new FlowLayout());
            //this.add(new SimuWindow());
         
            
            
            
            this.setVisible(true);
            this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            /*
             *  final JDesktopPane theDesktop = new JDesktopPane();
			 *	getContentPane().add(theDesktop);
             */
	}//public MainWIndow() {
    	
	    public boolean checkLibrary() {
			String javaLibPath = System.getProperty("java.library.path");
			String library = System.mapLibraryName("CellularAutomata-A-2D");
			File f = new File(javaLibPath + "/" + library);
			if (!(f.exists() && f.isFile())) {
				return false;
			}
			return true;
		}
    	
		private void createMenu() {
    		mMenuHash = new int[ITENS];
            int index = 0;
            
            MenuEvent menuEvent = new MenuEvent();
            mMenuBar = new JMenuBar();
            mMenu    = new JMenu("Simulation");
            mMenuBar.add(mMenu);

            /*
            mMenuItem = new JMenuItem("Simulation");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[index++] = mMenuItem.hashCode();
            mMenu.add(mMenuItem);
			*/
//--------------------------------------------------------------------
            mSubmenu = new JMenu("New");

            mMenuItem = new JMenuItem("Game of life");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[GAME_OF_LIFE] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            
            mMenuItem = new JMenuItem("Prisoner's Dilemma");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[PRISONERS_DILEMMA] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            
            mMenuItem = new JMenuItem("Predator-Prey");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[PREDATOR_PREY] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            
            mMenuItem = new JMenuItem("Segregation two-classes");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[SEGREGATION_2] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            
            mMenu.add(mSubmenu);
  //--------------------------------------------------------------------
            mMenuItem = new JMenuItem("Load native code simulation");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[NATIVE_SIMULATION] = mMenuItem.hashCode();
            mMenu.add(mSubmenu);
            /*mMenuItem = new JMenuItem("Native based on GPU");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[NATIVE_ON_GPU] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            
            mMenuItem = new JMenuItem("Native based on CPU");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[NATIVE_ON_CPU] = mMenuItem.hashCode();
            mSubmenu.add(mMenuItem);
            */
            mMenuItem.setEnabled(checkLibrary());
            mMenu.add(mMenuItem);
//--------------------------------------------------------------------
            mMenuItem = new JMenuItem("Load from python class");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[LOAD_MODEL_PY] = mMenuItem.hashCode();
            mMenu.add(mMenuItem);

            /*
            mMenuItem = new JMenuItem("Save");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[SAVE] = mMenuItem.hashCode();
            mMenu.add(mMenuItem);

            mMenuItem = new JMenuItem("Save as");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[SAVE_AS] = mMenuItem.hashCode();
            mMenu.add(mMenuItem);
            */
            mMenu.addSeparator();
            mMenuItem = new JMenuItem("Exit");
            mMenuItem.addActionListener(menuEvent);
            mMenuHash[EXIT_CLOSE] = mMenuItem.hashCode();
            mMenu.add(mMenuItem);

            mMenu    = new JMenu("About");
            mMenuHash[ABOUT] = mMenuItem.hashCode();
            mMenuBar.add(mMenu);
            
            this.setJMenuBar(mMenuBar);
    	}
    	
    	private void call_game_of_life() {
    	   	NumberFormat format = NumberFormat.getInstance();
    	   	format.setGroupingUsed(false);
    	    NumberFormatter formatterx = new NumberFormatter(format);
    	    formatterx.setValueClass(Integer.class);
    	    formatterx.setMinimum(0);
    	    formatterx.setMaximum(Integer.MAX_VALUE);
    	    formatterx.setAllowsInvalid(false);
    	    // If you want the value to be committed on each keystroke instead of focus lost
    	    formatterx.setCommitsOnValidEdit(true);

    	    NumberFormatter formattery = new NumberFormatter(format);
    	    formattery.setValueClass(Integer.class);
    	    formattery.setMinimum(0);
    	    formattery.setMaximum(Integer.MAX_VALUE);
    	    formattery.setAllowsInvalid(false);
    	    // If you want the value to be committed on each keystroke instead of focus lost
    	    formattery.setCommitsOnValidEdit(true);

    	    
    	    JFormattedTextField x_axis = new JFormattedTextField(formatterx);
    	    JFormattedTextField y_axis = new JFormattedTextField(formattery);
			x_axis.setText("100");
			y_axis.setText("80");
			
			//JTextField x_axis = new JTextField("100");
			//JTextField y_axis = new JTextField("80");
			String s1[] = {"periodic", "reflexive", "constant [0]", "constant [1]"};
			JComboBox<String> combo = new<String>JComboBox<String>(s1);
			 
			
			
			JComponent[] inputs = new JComponent[] {
			        new JLabel("x-axis lenght"),x_axis,
			        new JLabel("y-axis lenght"),y_axis,
			        new JLabel("Boundary condition"), combo
			};
			
			
			int result = JOptionPane.showConfirmDialog(null, inputs, "Game of Life - SETUP", JOptionPane.PLAIN_MESSAGE);
			if (result == JOptionPane.OK_OPTION) {
			    System.out.println("Your options:" +
			    		x_axis.getText() + ", " +
			    		y_axis.getText() + ", " +
			    		String.valueOf(combo.getSelectedItem()));
			    
			    SimuWindow s = new SimuWindow(GAME_OF_LIFE, 
			    		Integer.valueOf(x_axis.getText()), 
			    		Integer.valueOf(y_axis.getText()), 
			    		String.valueOf(combo.getSelectedItem()), mPainel);
			  
            	mPainel.add(s);
			
			} else {
			    System.out.println("User canceled / closed the dialog, result = " + result);
			}

    	}
    	
    	public void call_segregation_2() {
    		JFileChooser fileChooser = new JFileChooser();
    		FileNameExtensionFilter filter = new FileNameExtensionFilter("JSON config", "json", "JavaScript Object Notation");
        	fileChooser.setFileFilter(filter);
        	fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
        	if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
        	    // user selects a file
        		System.out.println("Here, Get json file so that Segregation module can read");
        		
        	}
    		/*
    	   	NumberFormat format = NumberFormat.getInstance();
    	   	format.setGroupingUsed(false);
    	    NumberFormatter formatterx = new NumberFormatter(format);
    	    formatterx.setValueClass(Integer.class);
    	    formatterx.setMinimum(0);
    	    formatterx.setMaximum(Integer.MAX_VALUE);
    	    formatterx.setAllowsInvalid(false);
    	    // If you want the value to be committed on each keystroke instead of focus lost
    	    formatterx.setCommitsOnValidEdit(true);

    	    NumberFormatter formattery = new NumberFormatter(format);
    	    formattery.setValueClass(Integer.class);
    	    formattery.setMinimum(0);
    	    formattery.setMaximum(Integer.MAX_VALUE);
    	    formattery.setAllowsInvalid(false);
    	    // If you want the value to be committed on each keystroke instead of focus lost
    	    formattery.setCommitsOnValidEdit(true);

    	    
    	    JFormattedTextField x_axis = new JFormattedTextField(formatterx);
    	    JFormattedTextField y_axis = new JFormattedTextField(formattery);
			x_axis.setText("10");
			y_axis.setText("10");
			
			//JTextField x_axis = new JTextField("100");
			//JTextField y_axis = new JTextField("80");
			String s1[] = {"periodic", "reflexive", "constant [0]", "constant [1]"};
			JComboBox<String> combo = new<String>JComboBox<String>(s1);
			 
			
			
			JComponent[] inputs = new JComponent[] {
			        new JLabel("x-axis lenght"),x_axis,
			        new JLabel("y-axis lenght"),y_axis,
			        new JLabel("Boundary condition"), combo
			};
			
			
			int result = JOptionPane.showConfirmDialog(null, inputs, "Segregation 2 groups", JOptionPane.PLAIN_MESSAGE);
			if (result == JOptionPane.OK_OPTION) {
			    System.out.println("Your options:" +
			    		x_axis.getText() + ", " +
			    		y_axis.getText() + ", " +
			    		String.valueOf(combo.getSelectedItem()));
			    
			    SimuWindow s = new SimuWindow(SEGREGATION_2, 
			    		Integer.valueOf(x_axis.getText()), 
			    		Integer.valueOf(y_axis.getText()), 
			    		String.valueOf(combo.getSelectedItem()), mPainel);
			  
            	mPainel.add(s);
			
			} else {
			    System.out.println("User canceled / closed the dialog, result = " + result);
			}
			*/
/*
    		SimuWindow s = new SimuWindow(SEGREGATION_2, 
			    		Integer.valueOf(x_axis.getText()), 
			    		Integer.valueOf(y_axis.getText()), 
			    		String.valueOf(combo.getSelectedItem()), mPainel);
			  
		    SegregationModel model = (SegregationModel) s.getModel();
*/	 
    	}//public void call_segregation_2() {
    	

    	
    	/*
    	 * Class listens menu click event
    	 */
        private class MenuEvent implements ActionListener{

            @Override
            public void actionPerformed(ActionEvent e) {
            	JFileChooser fileChooser = null;
            	FileNameExtensionFilter filter = null;
                int option = -1;
                for (int i = 0; i < ITENS; i++){
                    if (e.getSource().hashCode() == mMenuHash[i]){
                        option = i;
                        i = ITENS + 1;
                    }
                }
                switch (option){
                /*
                case 0: 
                	SimuWindow s = new SimuWindow();
                	mPanel.add(s);break;
                	*/
                case NATIVE_SIMULATION:
                	fileChooser = new JFileChooser();
                	filter = new FileNameExtensionFilter("JSON config", "json", "JavaScript Object Notation");
                	fileChooser.setFileFilter(filter);
                	fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
                	if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                	    // user selects a file
                		System.out.println("Here, pass to C native code the JSON file name so that C can open it");
                		
                	}
                	break;
                    //case 0:JOptionPane.showMessageDialog(null,"New","TITULO", JOptionPane.INFORMATION_MESSAGE);;break;
                case LOAD:JOptionPane.showMessageDialog(null,"LOAD","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                //case SAVE:JOptionPane.showMessageDialog(null,"SAVE","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                //case SAVE_AS:JOptionPane.showMessageDialog(null,"SAVE_AS","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                case LOAD_MODEL_PY:
                	fileChooser = new JFileChooser();
                	filter = new FileNameExtensionFilter("PYTHON SCRIPT", "py", "python	");
                	fileChooser.setFileFilter(filter);
                	fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
                	if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                	    // user selects a file
                		String f = fileChooser.getSelectedFile().getPath();
                		CAModelPy pyCA = new CAModelPy(f);
                	    SimuWindow s = new SimuWindow(pyCA, mPainel);
                    	mPainel.add(s);
                		
                	}
                	
                	break;
                case EXIT_CLOSE:System.exit(0);
                case ABOUT:JOptionPane.showMessageDialog(null,"ABOUT","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                case SEGREGATION_2:call_segregation_2();break;
                case GAME_OF_LIFE:call_game_of_life();break;
                case PRISONERS_DILEMMA:JOptionPane.showMessageDialog(null,"PRISONERS_DILEMMA","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                case PREDATOR_PREY:JOptionPane.showMessageDialog(null,"PREDATOR_PREY","TITULO", JOptionPane.INFORMATION_MESSAGE);break;
                    //case 4:JOptionPane.showMessageDialog(null,"MENSAGEM AQUI","TITULO", JOptionPane.INFORMATION_MESSAGE);;break;

                }
                /*
	private static final int EXIT_CLOSE		= 0;
	private static final int LOAD 			= 1;
	private static final int SAVE       	= 2;
	private static final int SAVE_AS  		= 3;
	private static final int ABOUT          = 4;
	private static final int GAME_OF_LIFE   = 5;
	private static final int PRISONERS_DILEMMA = 6;
	private static final int PREDATOR_PREY     = 7;
            	*/
            }//end-public void actionPerformed(ActionEvent e) {


        }//end- private class MenuEvent implements ActionListener{	
}//public class MainWIndow  extends JFrame {
