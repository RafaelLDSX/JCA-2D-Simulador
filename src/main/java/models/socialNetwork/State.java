package models.socialNetwork;

public enum State {

	ZEROSTATE(0),
	SUSCEPTIBLE(1),
	EXPOSED(2),
	INFECTED(3),
	RECOVERED(4);
	
	private int value;
	
	private State(int value) {
		this.value = value;
	}
	
	public int getValue() {
		return this.value;
	}
	
}
