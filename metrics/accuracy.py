from metrics.base_metric import BaseMetric

class Accuracy(BaseMetric):
    def __init__(self, normalize=True, name='accuracy'):
        super().__init__(name)

        self.normalize = normalize
    

    def compute_metric(self):
        if self.normalize:
            accuracy = self._num_corrects / self._num_samples
        else:
            accuracy = self._num_corrects
        return accuracy