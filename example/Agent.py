from stable_baselines import DQN

#use the saved model to form an agent
class Agent:
    model = DQN.load("best_model.pkl")

    def __init__(self):
        self.name = "DQNAgent"

    def act(self, stateObs, actions):
        action, _states = self.model.predict(stateObs)
        return action
