
__all__ = ['EarlyStopping']
class EarlyStopping:
    def __init__(self, patience = 30, min_delta = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, model, validation_loss):
        if validation_loss < 0:
            self.min_delta = -self.min_delta
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_weights)
                return True, self.min_validation_loss
        return False, None