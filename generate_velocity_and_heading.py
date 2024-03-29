import numpy as np

class Agent:
    def __init__(self, time, position, velocity, heading):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.heading = heading

def create_agents(positions, time_diffs):
    """
    Create a list of Agent objects based on positions and time differences.

    Args:
        positions (list): List of top down (x, y) positions of the object.
        time_diffs (list): List of time differences between consecutive positions.

    Returns:
        list: List of Agent objects with time, position, velocity, and heading.
    """
    agents = []
    cumulative_time = 0.0

    for i in range(len(positions)):
        position = np.array(positions[i])

        if i == 0:
            # First position, set default velocity and heading
            velocity = np.array([0.0, 0.0])
            heading = 0.0
        else:
            # Calculate velocity and heading
            prev_position = np.array(positions[i - 1])
            time_diff = time_diffs[i - 1]
            velocity = (position - prev_position) / time_diff
            heading = np.degrees(np.arctan2(velocity[1], velocity[0]))

        agent = Agent(cumulative_time, position, velocity, heading)
        agents.append(agent)

        if i < len(time_diffs):
            cumulative_time += time_diffs[i]

    return agents

# Example usage
positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
time_diffs = [0.1, 0.1, 0.1]

agents = create_agents(positions, time_diffs)

for agent in agents:
    print(f"Time: {agent.time:.1f}, Position: {agent.position}, Velocity: {agent.velocity}, Heading: {agent.heading:.2f}")