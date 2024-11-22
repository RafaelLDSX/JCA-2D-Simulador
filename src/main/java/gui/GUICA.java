	package gui;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;

import javax.swing.JPanel;


//https://stackoverflow.com/questions/33937377/how-to-draw-on-jpanel-with-jogl
public class GUICA extends JPanel{
	
	
	private float 	mScaleX = 0.0f,
					mScaleY = 0.0f,
					mWidth = 0.0f,
				    mHeight = 0.0f;
	
	private models.SimpleCAModel mCA = null;
	private boolean mHasCA = false;
	
	private final Color COLOR[] = new Color[6];
	
	public void setCellularAutomataModel(models.SimpleCAModel ca) {
		mCA = ca;
		if (mCA != null)
			mHasCA = true;
		else
			mHasCA = false;
		loadMesh();
		
	}//public void setCellularAutomataModel(models.SimpleCAModel ca) {
	 
	public void loadMesh() {
		if (mHasCA) {
			float w = (float) mCA.getWidth();
			float h = (float) mCA.getHeight();
			mScaleX = mWidth / w;
			mScaleY = mHeight / h;	
		}
		repaint();
	}//public void loadMesh() {
	public GUICA() {
		super();
		/*Dimension d = new Dimension(mWidth+3, mHeight+3);
		setSize(d);
		setMinimumSize(d);
		setMaximumSize(d);
		*/
		
		mWidth = 858.0f;  mHeight = 530.0f;
		//mCA = new models.CellularAutomataModel();
		mHasCA = false;
		 
		setPreferredSize(new Dimension((int)mWidth, (int)mHeight));
		setBackground(Color.BLACK);
		setAlignmentX(Component.CENTER_ALIGNMENT);
		//Window event listener
		addComponentListener(new MyJPanelEvent());
		addMouseListener(new MyMouseListener());
		
		
		//set color in according to state
		//vector color index is given by state
		COLOR[0] = Color.red;
		COLOR[1] = Color.green;
		COLOR[2] = Color.blue;
		COLOR[3] = Color.yellow;
		COLOR[4] = Color.orange;
		COLOR[5] = Color.white;
		
		

	}//public GuiCA() {
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        
        drawGrid( (Graphics2D) g);
    }//protected void paintComponent(Graphics g) {

    private void drawGrid(Graphics2D g2d) {
    	g2d.setBackground(Color.black);
    	if (!mHasCA) return;
 
    	for (int j = 0; j < mCA.getHeight(); j++) {
    		for (int i = 0; i < mCA.getWidth(); i++) {
    			int s = mCA.getStateCell(i, j);
    			if (s > 0) {
    				g2d.setColor(COLOR[s]);
    				g2d.fill(new Rectangle2D.Float(i * mScaleX, j * mScaleY, mScaleX, mScaleY ));
    				
    				/*
    				float sx9 = mScaleX * 0.8f;
    				float sy9 = mScaleY * 0.8f;
    				float sx1 = (mScaleX * 0.2f)/2.0f;
    				float sy1 = (mScaleY * 0.2f)/2.0f;
    				
    				g2d.fill(new Rectangle2D.Float(i * sx9 + sx1, j * sy9 + sy1, sx9, sy9 ));
					*/
    			}
    				
    		}//for (int i = 0; i < mCA.getWidth(); i++) {
    	}//for (int j = 0; j < mCA.getHeight(); j++) {
    	
//------------------------------------------------------------------------------------------------------------
    	
    	
    	float dh = 0.0f;
    	g2d.setColor(Color.YELLOW);
    	while (dh <=  mWidth) {
    		g2d.draw(new Line2D.Float( dh, 0.0f, dh,mHeight)) ;
    		dh += mScaleX;
    	}

    	dh = 0.0f;
    	while (dh <= mWidth) {
    		g2d.draw(new Line2D.Float( 0.0f, dh, mWidth, dh )) ;
    		dh += mScaleY;
    	}

    	

    }//drawgrid
    /*
    public void updatePanelSize(int w, int h) {
    	mWidth = w; mHeight = h;
    	mScaleX = mWidth / (float) mCA.getX();
		mScaleY = mHeight / (float) mCA.getY();
	
    	repaint();
    	
        GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice();
        float monitorWidth = gd.getDisplayMode().getWidth();
        float monitorHeight = gd.getDisplayMode().getHeight();

        // Aspect ratio of the monitor in decimal form.
        float monitorRatio = monitorWidth / monitorHeight;

        JComponent parent = (JComponent) getParent();
        mWidth = parent.getWidth();
        mHeight = parent.getHeight();

        mWidth = Math.min(mWidth, mHeight * monitorRatio);
        mHeight = mWidth / monitorRatio;

        // I am subtracting the width and height by their respected aspect ratio
        // coefficients (1920x1080 -> 16:9 (width:height)) and multiplying them 
        // by some scale (in this case 10) to add a "padding" to the JPanel.
        // The ratio coefficients and scale will need to be edited based upon the
        // resolution of your monitor.
        setPreferredSize(new Dimension((int)mWidth - (16 * 10), (int)mHeight - (9 * 10)));
        
        this.repaint();

        System.out.println("PanelRes: " + ((int)mWidth - (16 * 10)) + "x" + ((int)mHeight - (9 * 10)));
        System.out.println("PanelRatio: " + getWidth() / getHeight());
       
    } */
    
    private class MyMouseListener implements MouseListener{

		@Override
		public void mouseClicked(MouseEvent arg0) {
			// TODO Auto-generated method stub
			System.out.println("mouseClicked:" + arg0.getX() + "," + arg0.getY());
			System.out.println("            :" + mWidth + "," + mHeight);
		}

		@Override
		public void mouseEntered(MouseEvent arg0) {
			// TODO Auto-generated method stub
			//System.out.println("mouseEntered");
		}

		@Override
		public void mouseExited(MouseEvent arg0) {
			// TODO Auto-generated method stub
			//System.out.println("mouseExited");
		}

		@Override
		public void mousePressed(MouseEvent arg0) {
			// TODO Auto-generated method stub
			//System.out.println("mousePressed");
		}

		@Override
		public void mouseReleased(MouseEvent arg0) {
			// TODO Auto-generated method stub
			//System.out.println("mouseReleased");
		}
    	
    }
    private class MyJPanelEvent implements ComponentListener{

		@Override
		public void componentResized(ComponentEvent e) {
			// TODO Auto-generated method stub
			mWidth = e.getComponent().getSize().width;
			mHeight = e.getComponent().getSize().height;
			
			if (mHasCA) {
				mScaleX = mWidth / (float) mCA.getWidth();
				mScaleY = mHeight / (float) mCA.getHeight();	
			}
	    	
			
		
	    	repaint();
		}

		@Override
		public void componentMoved(ComponentEvent e) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void componentShown(ComponentEvent e) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void componentHidden(ComponentEvent e) {
			// TODO Auto-generated method stub
			
		}
    	
    }
}//public class GuiCA extends JPanel{