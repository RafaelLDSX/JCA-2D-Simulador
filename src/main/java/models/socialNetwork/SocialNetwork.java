package models.socialNetwork;

import models.CellularAutomataModel;

public class SocialNetwork extends CellularAutomataModel{

	private double alpha;
	private double averageAlpha;
	private double gamma;
	private double averageGamma;
	private double epsilon;
	private double averageEpsilon;
	
	private double alphaRumorMI;
	private double alphaRumorSIG;
	private double gammaRumorMI;
	private double gammaRumorSIG;
	
	private double maxOutDegree;
	private double averageOutDegree;
	private double maxInDegree;
	private double averageInDegree;
	
	
	public SocialNetwork() {
		
		this.maxCellStates = 5;
		
	}
	
}
