package models.socialNetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
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
	
	private List<Agent> agents;
	
	
	public SocialNetwork() {
		
		this.maxCellStates = 5;
		this.agents = new ArrayList<Agent>();
		
	}
	
	@Override
	public void initialCondition() {
		System.out.println("initial condition");
		this.state = CLASS_STATE_INITIAL_CONDITION;
		double uniform = ThreadLocalRandom.current().nextDouble(0.0, 1.0);
		
		double normalSample = ThreadLocalRandom.current().nextGaussian();
        double alphaRumor = Math.exp(this.alphaRumorMI + this.alphaRumorSIG * normalSample);
        normalSample = ThreadLocalRandom.current().nextGaussian();
        double gammaRumor = Math.exp(this.gammaRumorMI + this.gammaRumorSIG * normalSample);
        
        Graph<Agent, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);
        int numberOfVertices;
        
        try (BufferedReader reader = new BufferedReader(new FileReader("BA-N-10-P-1-C-6.net"))) {
            String line;
            line = reader.readLine();
            if (line.startsWith("*Vertices")) {
                // separa a string por espaços
                String[] tokens = line.split("\\s+");
                if (tokens.length > 1) {
                    numberOfVertices = Integer.parseInt(tokens[1]);
                    for (int i = 0; i < numberOfVertices; i++) {
                    	Agent x = new Agent();
                    	this.agents.add(x);
                    	graph.addVertex(x);
                    }
                }
            }
            line = reader.readLine();
            if (line.startsWith("*Arcs")) {
            	Agent auxAgent1, auxAgent2;
            	while ((line = reader.readLine()) != null) {
            		String[] indexes = line.split("\\s+");
            		auxAgent1 = this.agents.get(Integer.parseInt(indexes[0]) - 1);
            		auxAgent2 = this.agents.get(Integer.parseInt(indexes[1]) - 1);
            		graph.addEdge(auxAgent1, auxAgent2);		
            		
            	}
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        Agent maxInDegreeAgent = agents.stream().max(Comparator.comparing(x -> graph.inDegreeOf(x))).get();
        Agent maxOutDegreeAgent = agents.stream().max(Comparator.comparing(x -> graph.outDegreeOf(x))).get();
        this.maxInDegree = graph.inDegreeOf(maxInDegreeAgent);
        this.maxOutDegree = graph.outDegreeOf(maxOutDegreeAgent);
        
        double totalInDegree = agents.stream().collect(Collectors.summingInt(x -> graph.inDegreeOf(x)));
        double totalOutDegree = agents.stream().collect(Collectors.summingInt(x -> graph.outDegreeOf(x)));
        this.averageInDegree = totalInDegree / agents.size();
        this.averageOutDegree = totalOutDegree / agents.size();
        
        double[] probabilities = new double[agents.size()];
        double add = 0;
        for(int i = 0; i < agents.size(); i++) {
        	add += graph.inDegreeOf(agents.get(i)) / totalInDegree;
        	probabilities[i] = add;
        }
        
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
