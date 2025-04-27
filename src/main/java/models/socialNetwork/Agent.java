package models.socialNetwork;

import java.util.ArrayList;
import java.util.List;

public class Agent {

	public Agent() {
		this.id = 0;
		this.state0 = State.ZEROSTATE;
		this.state1 = State.ZEROSTATE;
		this.inDegree = 0;
		this.outDegree = 0;
		this.inAgents = new ArrayList<Agent>();
		this.outAgents = new ArrayList<Agent>();
	}
	
	private int id;
	private State state0;
	private State state1;
	private int inDegree;
	private int outDegree;
	private List<Agent> inAgents;
	private List<Agent> outAgents;
	
}
