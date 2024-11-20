package gui;


import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JDesktopPane;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JPopupMenu;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import javax.swing.ListSelectionModel;
import javax.swing.border.TitledBorder;
import javax.swing.event.InternalFrameAdapter;
import javax.swing.event.InternalFrameEvent;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.text.NumberFormatter;

import org.python.modules.synchronize;


import models.CAModelPy;
import models.SimpleCAModel;

import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;


public class SimuWindow extends JInternalFrame{
	private static final int ITENS  			= 6;
	private static final int NEXT_STEP 			= 0;
	private static final int RUN_CA       		= 1;
	private static final int MODEL_CA_BG	    = 2;
	private static final int INIT_CA     	    = 3;
	private static final int SAVE_STATE     	= 4;
	private static final int LOAD_STATE     	= 5;
	private SimuWindow mPtr = null;
	/*
	private JButton mNext = null;
	private JButton mPrevious = null;
	private JButton mSaveSnapshot = null;
	*/
	private int []mMenuHash;
    private JMenuBar mMenuBar = null;
    private JMenuItem mMenuItem = null;
    private JMenu     mMenu     = null;
                     // mSubmenu  = null;
    
    private JList<String>mLayerList  = null;
    private DefaultListModel mLayerModel = null;
    //private Vector<Layer>mLayers = null;
    private GUICA mGUICA = null;
    private models.SimpleCAModel mCA = null;
    																																																																																																														
    
    private BGThread mThreadBG = null;
    private FGThread mThreadFG = null;
    private JDesktopPane mPainel =  null;
    
    private void closeFrameEvent() {
    	 addInternalFrameListener(new InternalFrameAdapter(){
             public void internalFrameClosing(InternalFrameEvent e) {
            	 mPtr.dispose();
            	 //mPainel.remove(mPtr);
             }
         });
    }

    public SimuWindow(CAModelPy capy, JDesktopPane p ) {
		super(capy.getTitle(), true, true, true, false);
		this.mPainel = p;
		mPtr = this;
		closeFrameEvent();
		createMenu(); 
        //Adding GUI JPanels
        this.getContentPane().add(createLayer(), BorderLayout.LINE_START);
        this.getContentPane().add(createLattice(), BorderLayout.CENTER);
        //this.pack();
		this.setLocation(30, 30);
		this.setSize(1046, 700);
		this.setVisible(true);
		this.show();
		 
		
		mCA = capy;
    	mGUICA.setCellularAutomataModel(mCA);
	}//public SimuWindow() {    

    public SimuWindow(String modelname, JDesktopPane p) {
    	super(modelname, true, true, true, false);
    	this.mPainel = p;
		mPtr = this;
		closeFrameEvent();
        createMenu();
    	Settings s = new Settings(modelname);
    	
    	//AQUI
    }
    
    public SimuWindow(int type, int w, int h, String boundary, JDesktopPane p) {
		super("Cellular Automata Simulation", true, true, true, false);
		this.mPainel = p;
		mPtr = this;
		closeFrameEvent();
        createMenu(); 
        //Adding GUI JPanels
        this.getContentPane().add(createLayer(), BorderLayout.LINE_START);
        this.getContentPane().add(createLattice(), BorderLayout.CENTER);
        //this.pack();
		this.setLocation(30, 30);
		this.setSize(1046, 700);
		this.setVisible(true);
		this.show();
		 
		//mLayers = new Vector<Layer>();
		
		if (type == MainWindow.GAME_OF_LIFE) {
			mCA = new models.GameOfLife(w, h, boundary);
	    	mCA.initialCondition();
	    	for (int i = 0; i < mCA.getLayersSize(); i++)
	    		mLayerModel.addElement(new String(mCA.getLayerName(i)));
	    	mGUICA.setCellularAutomataModel(mCA);
	    	setTitle("Game of life model");	
	    	return;
		}
		
		if (type == MainWindow.SEGREGATION_2) {
			mCA = new models.SegregationModel(w, h, boundary);
	    	mCA.initialCondition();
	    	for (int i = 0; i < mCA.getLayersSize(); i++)
	    		mLayerModel.addElement(new String(mCA.getLayerName(i)));
	    	mGUICA.setCellularAutomataModel(mCA);
	    	setTitle("Segregation model");
	    	return;
		}
	}//public SimuWindow() {    
    
    public models.SimpleCAModel getModel() { return mCA; }
    /*
     * Creates menu components
     */
    private void createMenu() {
		mMenuHash = new int[ITENS];
         
        
        MenuEvent menuEvent = new MenuEvent();
        mMenuBar = new JMenuBar();
//--------------------------------------------------------------------------------------------------        
        mMenu    = new JMenu("Commands");
        mMenuBar.add(mMenu);
 
        mMenuItem = new JMenuItem("initial condition");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F4, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[INIT_CA] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);// index = 3

        mMenuItem = new JMenuItem("run fg/stop");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F5, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[RUN_CA] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);// index = 3

        mMenuItem = new JMenuItem("run bg/stop");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F6, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[MODEL_CA_BG] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);// index = 3

        mMenuItem = new JMenuItem("next step");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F7, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[NEXT_STEP] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);
        
        mMenuItem = new JMenuItem("load");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F9, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[LOAD_STATE] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);
        
        mMenuItem = new JMenuItem("save");
        mMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F8, 0));
        mMenuItem.addActionListener(menuEvent);
        mMenuHash[SAVE_STATE] = mMenuItem.hashCode();
        mMenu.add(mMenuItem);
        this.setJMenuBar(mMenuBar);
//--------------------------------------------------------------------------------------------------
	}
	
    /*
     * Creates layer GUI component - It is a List
     * 
     */
    private JPanel createLayer() {
    	int w = 140,
    	    h = 530;
    	
    	mLayerModel =  new DefaultListModel();
    	
        mLayerList = new<String> JList();
		mLayerList.setModel(mLayerModel);
		
		Dimension d = new Dimension(w, h);
		mLayerList.setSize(d);
		mLayerList.setMinimumSize(d);
		mLayerList.setMaximumSize(d);
		mLayerList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		//mLayerList.addMouseListener(null);
		mLayerList.addMouseListener(new MouseAdapter() {
	        public void mousePressed(MouseEvent e) {
	            if (e.getButton() == MouseEvent.BUTTON3) {
	                JPopupMenu menu = new JPopupMenu();
	                JMenuItem item = new JMenuItem("Data");
	                
	                item.addActionListener(new ActionListener() {
	                    public void actionPerformed(ActionEvent e) {
	                    	SimuWindowData swd = new SimuWindowData(mCA.getLayerName(mLayerList.getSelectedIndex()));
	                    	mPainel.add(swd);
	                    	swd.toFront();
	                    	swd.setLog(mCA.getLogBasedOnLayer(mLayerList.getSelectedIndex()));
	                    	//System.out.println("Your option was: " + Integer.toString(mLayerList.getSelectedIndex()));
	                    }//public void actionPerformed(ActionEvent e) {
	                });// item.addActionListener(new ActionListener() {
	                menu.add(item);
	                //tem erro aqui, possivelmente colocar a Jframe pai deste item no lugar
	                
	                menu.show(mPtr, e.getX()+10, e.getY()+60);
	                //menu.show(this, 5, mLayerList.getCellBounds(
	                //		mLayerList.getSelectedIndex() + 1,
	                //		mLayerList.getSelectedIndex() + 1).y);
	            }
	        }
	    });
		
		/*
		mLayerList.addListSelectionListener(new ListSelectionListener() {

			@Override
			public void valueChanged(ListSelectionEvent event) {
				// TODO Auto-generated method stub
				if (!event.getValueIsAdjusting()){
					int a = event.getFirstIndex();
					int b = event.getLastIndex();
		            JList source = (JList)event.getSource();
		            System.out.println("!getValueIsAdjusting");
		            System.out.println(source.getSelectedValue().toString());
		            System.out.println(a);
		            System.out.println(b);
		        }else {
		        	int a = event.getFirstIndex();
					int b = event.getLastIndex();
		            JList source = (JList)event.getSource();
		            System.out.println("getValueIsAdjusting");
		            System.out.println(source.getSelectedValue().toString());
		            System.out.println(a);
		            System.out.println(b);
		        }
			}
			
		});
		*/
		LogPanel panel = new LogPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS));
		
		mLayerList.setAlignmentX(Component.CENTER_ALIGNMENT);
		panel.setBorder( BorderFactory.createTitledBorder(BorderFactory.createLineBorder(Color.black), "Layers/Attributes"));
		panel.add(Box.createVerticalStrut(2));
        panel.add(mLayerList);
        panel.add(Box.createVerticalStrut(2));
        d = new Dimension(w+5, h+5);
        panel.setSize(d);
        panel.setMinimumSize(d);
        panel.setMaximumSize(d);
        panel.setPreferredSize(d);

        d = null;
        return panel;
        
    }
    
    /*
     * Creates layer GUI components
     */
    private JPanel createLattice() {
    	JPanel panel = new JPanel();
    	
    	GUICA p1 = new GUICA();
    	mGUICA = p1;
		//mGUIA.setCellularAutomataModel(new models.CellularAutomataModel());
		panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS));
		panel.setBorder( BorderFactory.createTitledBorder(BorderFactory.createLineBorder(Color.black), "Lattice"));
		panel.add(Box.createVerticalStrut(2));
        panel.add(p1);
        panel.add(Box.createVerticalStrut(2));

        //BoxLayout.PAGE_AXIS -> horizontal orientation
		//BoxLayout.LINE_AXIS -> vertical orientation
		return panel;
        
    }

    private void runSimulationBG() {
    	
    	if (mThreadBG != null) {
    		boolean ret =  mThreadBG.isAlive();
    		mThreadBG.setIsRunningFalse();
    		try {
    			mThreadBG.join();
    		} catch (InterruptedException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
    		mThreadBG = null;
    		if (ret) return;
    	}//if (mThreadBG != null) {
    	
    	NumberFormat format = NumberFormat.getInstance();
    	format.setGroupingUsed(false);
	    NumberFormatter formattert = new NumberFormatter(format);
	    formattert.setValueClass(Integer.class);
	    formattert.setMinimum(0);
	    formattert.setMaximum(Integer.MAX_VALUE);
	    formattert.setAllowsInvalid(false);
	    // If you want the value to be committed on each keystroke instead of focus lost
	    formattert.setCommitsOnValidEdit(true);

	    JFormattedTextField timesteps = new JFormattedTextField(formattert);
	    timesteps.setText("1000");
		
		JComponent[] inputs = new JComponent[] {
		        new JLabel("Timesteps: "),timesteps,
		};
		
		int result = JOptionPane.showConfirmDialog(null, inputs, "CA in background timesteps", JOptionPane.PLAIN_MESSAGE);
		if (result == JOptionPane.OK_OPTION) {
		    System.out.println("Your options:" + timesteps.getText());
		    int ts = Integer.parseInt(timesteps.getText());
		    mThreadBG = new BGThread(mCA, mGUICA, ts);
		    mThreadBG.start();
		} else {
		    System.out.println("User canceled / closed the dialog, result = " + result);
		    mThreadBG = null;
		}
    }//private void runSimulationBG() {
    
    
    private void runSimulationFG() {
    	if (mThreadFG == null) {
    		mThreadFG = new FGThread(mCA, mGUICA);
    		mThreadFG.start();
    	}else {
    		mThreadFG.setIsRunningFalse();
    		try {
    			mThreadFG.join();
    		} catch (InterruptedException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
    		mThreadFG = null;
    	}
    }//private void runSimulation() {
    
    /*! \class BGThread
		\brief BGThread computes Cellular Automata in background. It executes n steps before stopping
    */
    private class BGThread extends Thread{
    	protected boolean mIsRunning = false;
    	public synchronized void setIsRunningTrue() {mIsRunning = true;}
    	public synchronized void setIsRunningFalse() {mIsRunning = false;}
    	public synchronized boolean getIsRunning() {return mIsRunning;}
    	protected models.SimpleCAModel mCA = null;
    	protected GUICA mGUICA = null;
    	private int mTS = -1;
    	public BGThread(models.SimpleCAModel ca, GUICA guica, int ts) {
    		super();
    		this.mCA = ca;
    		this.mGUICA = guica;
    		this.mTS = ts;
    		
    	}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			this.setIsRunningTrue();
			int ts = 0;
			
			JFrame frame = new JFrame("Simulation progress");
			JProgressBar progressbar = new JProgressBar(0, this.mTS);
			progressbar.setStringPainted(true);
			frame.add(progressbar);
		    
		    frame.setSize(400, 60);
		    frame.setVisible(true);
		    Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		    
		    int w = (int)(screenSize.getWidth() / 2) - 200;
		    int h = (int)(screenSize.getHeight() / 2) - 30;
		    frame.setResizable(false);
		    frame.setLocation(w, h);
		    
		    
			while(this.getIsRunning() && ts < this.mTS) {
				this.mCA.update();
				progressbar.setValue(ts);
				ts++;
			}

			progressbar.setValue(ts);
			frame.setVisible(false);
			frame = null;
			this.mGUICA.repaint();
		}
     
    }//private class BGThread extends Thread{
    
    /*! \class FGThread
		\brief FGThread computes Cellular Automata in foreground. It updates JPanel lattice state for each time step
    */
    private class FGThread extends BGThread{
    	
		public FGThread(SimpleCAModel ca, GUICA guica) {
    		super(ca, guica, 0);
    		
    	}
	
		@Override
		public void run() {
			this.setIsRunningTrue();
			while(this.getIsRunning()) {
				this.mCA.update();
        		this.mGUICA.repaint();
        		//System.out.println(i);
        		try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}//public void run() {
    	
    }//private class FGThread extends BGThread{

    /*! \class LogPanel
		\brief LogPanelEvent is used to personalize the window and allowing it to resize
    */
    private class LogPanel extends JPanel implements ComponentListener{

    	public LogPanel() {
    		this.addComponentListener(this);
    	}
		@Override
		public void componentHidden(ComponentEvent arg0) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void componentMoved(ComponentEvent arg0) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void componentResized(ComponentEvent arg0) {
			// TODO Auto-generated method stub
			Component c = (Component)arg0.getSource();
			System.out.println("LogPanel");
			System.out.println(c.getWidth());
			System.out.println(c.getHeight());
			Dimension d = new Dimension(c.getWidth()-12, c.getHeight()-30);
			mLayerList.setSize(d);
			mLayerList.setMinimumSize(d);
			mLayerList.setMaximumSize(d);
			
		}

		@Override
		public void componentShown(ComponentEvent arg0) {
			// TODO Auto-generated method stub
			
		}
    	
    }
    
    /*! \class MenuEvent
    	\brief MenuEvent is a menu event class of this SimuWindow. Each menu entry calls SimuWindow method.
    */
    private class MenuEvent implements ActionListener{

        @Override
        public void actionPerformed(ActionEvent e) {
            
            int option = -1;
            for (int i = 0; i < ITENS; i++){
                if (e.getSource().hashCode() == mMenuHash[i]){
                    option = i;
                    i = ITENS + 1;
                }
            }
    
            switch (option){
            
                
                case NEXT_STEP://	JOptionPane.showMessageDialog(null,"next step","TITULO", JOptionPane.INFORMATION_MESSAGE);
                		if (mCA != null) {
                			mCA.update();
                    		mGUICA.repaint();	
                		}
                		
                	break;
                case RUN_CA:
                	if (mCA != null) {
                		runSimulationFG();	
            		}
                	break;
                	
                case INIT_CA:
                	if (mCA != null) {
                		mCA.initialCondition();
                		mGUICA.loadMesh();
                		mGUICA.repaint();
            		}
                	
                	break;
                case MODEL_CA_BG:
                	runSimulationBG();
                	/*
                	mCA = new models.GameOfLife();
                	mCA.initialCondition();
                	mGUICA.setCellularAutomataModel(mCA);
                	setTitle("Game of life model");
                	*/
                	break;
                	
                case LOAD_STATE:
                	break;
                	
                case SAVE_STATE:
                	break;
                	
                //case 4:JOptionPane.showMessageDialog(null,"MENSAGEM AQUI","TITULO", JOptionPane.INFORMATION_MESSAGE);;break;

            }
            

        }//end-public void actionPerformed(ActionEvent e) {
    }//private class MenuEvent implements ActionListener{
}