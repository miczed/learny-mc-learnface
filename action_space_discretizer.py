

# This is how to discretize the action space
# After this we can deal with actions like "3" for accelerating
# the NN for example produces then five outputs (one for each action) rather than infinite for every continous combination

# by https://github.com/NotAnyMike/gym/blob/master/gym/envs/box2d/car_racing.py

# However I have no idea how to implement this into the envirnoment where it should be








    def _transform_action(self, action):
        if self.discretize_actions == "soft":
            raise NotImplementedError
        elif self.discretize_actions == "hard":
            # ("NOTHING", "LEFT", "RIGHT", "ACCELERATE", "BREAK")
            # angle, gas, break
            if action == 0: action = [ 0, 0, 0.0] # Nothing
            if action == 1: action = [-1, 0, 0.0] # Left
            if action == 2: action = [+1, 0, 0.0] # Right
            if action == 3: action = [ 0,+1, 0.0] # Accelerate
            if action == 4: action = [ 0, 0, 0.8] # break
            
            
            
    def step(self, action):
        action = self._transform_action(action)
        
        # ... the rest of the original step function follows
