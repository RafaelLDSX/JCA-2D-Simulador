package models.socialNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

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
	
	private double infected;
	private String infectedOrder;
	private int timesteps;
	
	
	public SocialNetwork() {
		
		this.maxCellStates = 5;
		
	}
	
	@Override
	public void initialCondition() {
		System.out.println("initial condition");
		this.state = CLASS_STATE_INITIAL_CONDITION;
		
		
	}
	
	public void loadConfig(String fileName) {
		 JSONParser parser = new JSONParser();
		 
		 try {
			 Object obj = parser.parse(new FileReader(fileName));
			 JSONObject globalParams = (JSONObject) obj;
			 JSONObject json = (JSONObject) globalParams.get("global-params");
			 
			 this.width = ((Long) json.get("cell-x")).intValue();
			 this.height = ((Long) json.get("cell-y")).intValue();
			 this.infected = (double) json.get("infected");
			 this.infectedOrder = (String) json.get("infected_init_order");
			 this.timesteps = ((Long) json.get("timesteps")).intValue();
			 this.alphaRumorMI = (double) json.get("alpha-rumor-mi");
			 this.alphaRumorSIG = (double) json.get("alpha-rumor-sig");
			 this.gammaRumorMI = (double) json.get("gamma-rumor-mi");
			 this.gammaRumorSIG = (double) json.get("gamma-rumor-sig");
			 
			 
		 } catch (FileNotFoundException e) {
			 //melhorar catch
			 e.printStackTrace();
		 } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
