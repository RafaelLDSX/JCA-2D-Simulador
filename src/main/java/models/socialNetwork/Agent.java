package models.socialNetwork;

import java.util.ArrayList;
import java.util.List;

public class Agent {

	public Agent() {
		this.id = 0;
		this.state0 = State.ZEROSTATE;
		this.state1 = State.ZEROSTATE;
		this.inAgents = new ArrayList<Agent>();
		this.outAgents = new ArrayList<Agent>();
	}
	
	public int id;
	public State state0;
	public State state1;
	public State state;
	public List<Agent> inAgents;
	public List<Agent> outAgents;
	
	public void setStates(State s) {
		this.state0 = s;
		this.state1 = s;
	}
	
	public State getState() {
		return this.state0;
	}
	
	public void setState(State s) {
		this.state1 = s;
	}
}
