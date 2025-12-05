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
import java.util.Random;
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

	private static double ERROR = 1E-20;
	private double alpha;
	private double averageAlpha;
	private double gamma;
	private double averageGamma;
	private double epsilon;
	private double averageEpsilon;
	private double lambda;
	
	private double alphaRumorMI;
	private double alphaRumorSIG;
	private double gammaRumorMI;
	private double gammaRumorSIG;
	
	private double maxOutDegree;
	private double averageOutDegree;
	private double maxInDegree;
	private double averageInDegree;
	
	private double infectionProbability;
	private String infectedOrder;
	private int timesteps;
	
	private Random generator = new Random(System.nanoTime());
	
	private Graph<Agent, DefaultEdge> graph;
	
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
        
        this.graph = new DefaultDirectedGraph<>(DefaultEdge.class);
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
                    	x.setStates(State.SUSCEPTIBLE);
                    	this.agents.add(x);
                    	this.graph.addVertex(x);
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
            		this.graph.addEdge(auxAgent1, auxAgent2);		
            		
            	}
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.alpha = this.averageAlpha = this.gamma = this.averageGamma = this.epsilon = this.averageEpsilon = 0;
        Agent maxInDegreeAgent = agents.stream().max(Comparator.comparing(x -> this.graph.inDegreeOf(x))).get();
        Agent maxOutDegreeAgent = agents.stream().max(Comparator.comparing(x -> this.graph.outDegreeOf(x))).get();
        this.maxInDegree = this.graph.inDegreeOf(maxInDegreeAgent);
        this.maxOutDegree = this.graph.outDegreeOf(maxOutDegreeAgent);
        
        double totalInDegree = agents.stream().collect(Collectors.summingInt(x -> this.graph.inDegreeOf(x)));
        double totalOutDegree = agents.stream().collect(Collectors.summingInt(x -> this.graph.outDegreeOf(x)));
        this.averageInDegree = totalInDegree / agents.size();
        this.averageOutDegree = totalOutDegree / agents.size();
        
        double[] probabilities = new double[agents.size()];
        double add = 0;
        for(int i = 0; i < agents.size(); i++) {
        	add += this.graph.inDegreeOf(agents.get(i)) / totalInDegree;
        	probabilities[i] = add;
        }
        
        double numberOfInfected;
        
        if (this.infectionProbability < 1.0) {
        	numberOfInfected = Math.floor(this.infectionProbability * agents.size());
        }
        else {
        	numberOfInfected = Math.floor(this.infectionProbability);
        }
        
        if (numberOfInfected == 0)
        	numberOfInfected = 1;
        
        if (this.infectedOrder == "rand") {
        	for (int i = 0; i < numberOfInfected; i++) {
        		boolean selected = false;
        		
        		do {
        			double p = generator.nextDouble();
            		add = probabilities[0];
            		int j = 1;
            		while (p > add) {
            			add = probabilities[j++];
            		}
            		j--;
            		if (agents.get(j).state0 == State.SUSCEPTIBLE) {
            			agents.get(j).setStates(State.INFECTED);
            			selected = true;
            		}
        		} while (!selected);
        		
        	}
        } else {
        	// fazer infecção por ordem crescente e decrescente
        }  
	}
	
	public void statistic(int i) {}
	
	public void exec() {
		int mark = this.timeStep / 10;
		
		for (int i = 1; i <= this.timeStep; i++) {
			this.update(i);
			this.statistic(i);
			if (i % mark == 0) {
			}
		}
	}
	
	public void update(int t) {
		for (int i = 0; i < this.agents.size(); i++) {
			switch (this.agents.get(i).getState()) {
				case SUSCEPTIBLE: this.susceptible2NS(i, t);
				case EXPOSED: this.exposed2NS(i, t);
				case INFECTED: this.infected2NS(i, t);
			}	
		}
		
		for (int i = 0; i < this.agents.size(); i++) {
			this.agents.get(i).update();
		}
	}
	
	double buildEpsilon(int node, int iTime) {
	    double dTime = (double) iTime;
	    double epsilon = 1.0 - Math.exp(-this.lambda * dTime);

	    if (Double.isNaN(epsilon)) {
	        throw new AssertionError("epsilon is NaN");
	    }
	    if (Double.isInfinite(epsilon)) {
	        throw new AssertionError("epsilon is Infinite");
	    }

	    return epsilon;
	}
	
	double buildAlpha(int node, int iTime) {
		double infected = 0.0;
		double recovered = 0.0;
		double alpha = 0.0;
		
		List<Agent> outAgents = this.agents.get(node).outAgents;
		
		for (Agent agent : outAgents) {
			if (agent.state0 == State.INFECTED) {
				infected += this.graph.inDegreeOf(agent);
			} else if (agent.state0 == State.RECOVERED) {
				recovered += this.graph.inDegreeOf(agent);
			}
		}
		
		double p1 = Math.exp(this.alphaRumorMI + this.alphaRumorSIG * generator.nextGaussian());

		double t1 = (double) iTime;
		
		if (iTime > 0)
			t1 /= p1;
		else
			t1 = ERROR / p1;
		
		alpha = (1.0 - Math.exp(-((infected / this.maxInDegree) * t1)));
		
		if (alpha < ERROR)
			alpha = 0;
		
		return alpha;
	}
	
	double buildGamma(int node, int iTime) {
		double infected = 0.0;
		double recovered = 0.0;
		double gamma = 0.0;
		
		List<Agent> outAgents = this.agents.get(node).outAgents;
		
		for (Agent agent : outAgents) {
			if (agent.state0 == State.RECOVERED) {
				recovered += this.graph.inDegreeOf(agent);
			}
		}
		
		double p1 = Math.exp(this.gammaRumorMI + this.gammaRumorSIG * generator.nextGaussian());
		
		double t1 = (double) iTime;
		
		if (iTime > 0)
			t1 /= p1;
		else
			t1 = ERROR / p1;
		
		gamma = (1.0 - Math.exp(-((infected / this.maxInDegree) * t1)));
		
		if (gamma < ERROR)
			gamma = 1.0 -Math.exp(-(double) iTime);
		
		return gamma;
	}
	
	void susceptible2NS(int node, int iTime) {
		double epsilon = this.buildEpsilon(node, iTime);
		double alpha = this.buildAlpha(node, iTime);
		double prob = generator.nextDouble();
		
		this.averageAlpha += alpha;
		this.alpha++;
		
		this.averageEpsilon += epsilon;
		this.epsilon++;
		
		double b = alpha + epsilon * (1.0 - alpha);
		
		if (prob < alpha)
			this.agents.get(node).setState(State.INFECTED);
		else if (alpha <= prob && prob < b)
			this.agents.get(node).setState(State.EXPOSED);
		else
			this.agents.get(node).setState(this.agents.get(node).getState());
		
	}
	
	void exposed2NS(int node, int iTime) {
		double gamma = buildGamma(node, iTime);
		double alpha = buildAlpha(node, iTime);
		
		//TODO if alpha < 0 exit(1)
		
		this.averageAlpha += alpha;
		this.alpha++;
		
		this.averageGamma += gamma;
		this.gamma++;
		
		double prob = generator.nextDouble();
		double b = gamma + (alpha * (1.0 - gamma));
		
		if (prob < gamma)
			this.agents.get(node).setState(State.RECOVERED);
		else if (gamma <= prob && prob <= b) 
			this.agents.get(node).setState(State.INFECTED);
		else 
			this.agents.get(node).setState(this.agents.get(node).getState());
	}
	
	void infected2NS(int node, int iTime) {
		double gamma = buildGamma(node, iTime);
		
		this.averageGamma += gamma;
		this.gamma++;
		
		double prob = generator.nextDouble();
		
		if (prob < gamma)
			this.agents.get(node).setState(State.RECOVERED);
		else
			this.agents.get(node).setState(this.agents.get(node).getState());
	}
	
	public void loadConfig(String fileName) {
		 JSONParser parser = new JSONParser();
		 
		 try {
			 Object obj = parser.parse(new FileReader(fileName));
			 JSONObject globalParams = (JSONObject) obj;
			 JSONObject json = (JSONObject) globalParams.get("global-params");
			 
			 this.width = ((Long) json.get("cell-x")).intValue();
			 this.height = ((Long) json.get("cell-y")).intValue();
			 this.infectionProbability = (double) json.get("infected");
			 this.infectedOrder = (String) json.get("infected_init_order");
			 this.timesteps = ((Long) json.get("timesteps")).intValue();
			 this.alphaRumorMI = (double) json.get("alpha-rumor-mi");
			 this.alphaRumorSIG = (double) json.get("alpha-rumor-sig");
			 this.gammaRumorMI = (double) json.get("gamma-rumor-mi");
			 this.gammaRumorSIG = (double) json.get("gamma-rumor-sig");
			 this.lambda = (double) json.get("lambda");
			 
			 
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
