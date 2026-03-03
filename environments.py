from main import *

class SwitchingAdversaryEnvironment(BanditEnvironment):
    def __init__(self, k=5, switch_at=150):
        super().__init__(k)
        self.switch_at = switch_at
        self.current_step = 0
        self.switched = False

    def reset(self):
        super().reset()
        self.current_step = 0
        self.switched = False
        # Ensure Arm 0 is the best at the start
        self.true_values[:] = 0
        self.true_values[0] = 1.0 

    def interact(self, action, prediction=None):
        # Prediction as unused arg to keep signature consistent
        self.current_step += 1
        
        # At switch_at, the 'best' arm moves to the other side
        if self.current_step > self.switch_at and not self.switched:
            self.true_values[0] = 0.0
            self.true_values[-1] = 1.0 # Move reward to the last arm
            self.switched = True
            
        return np.random.normal(self.true_values[action], 0.1)

    def get_best_reward(self):
        return 1.0 # The maximum reward is always 1.0