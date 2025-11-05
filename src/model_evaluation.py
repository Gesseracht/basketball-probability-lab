import numpy as np

class Evaluation:

    def __init__(self):
        pass
    
    def prediction(self):
        pass


    def evaluate_predictions(self, y_true, y_proba):
        # For each game:
            # 1. Calculate p_home with next_games()
            # 2. Retrieve the actual result (1 if home wins, 0 otherwise)
        return np.average(np.sum((y_true - y_proba) ** 2, axis=1))
