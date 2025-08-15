from metrics.base_metric import BaseMetric
import torch
class TopKAccuracy(BaseMetric):
    def __init__(self, top_k, normalize=True, name='accuracy'):
        super().__init__(name + '_top_' + str(top_k))
        self.normalize = normalize

    
    def update(self, model_output, y_true):
        y_pred_top5 = torch.topk(model_output['output'], 5, dim=1).indices
        for i in range(y_true.shape[0]):
            if y_true[i] in y_pred_top5[i]:
                self._num_corrects += 1
        self._num_samples += y_true.shape[0]

    
    def compute_metric(self):
        if self.normalize:
            accuracy = self._num_corrects / self._num_samples
        else:
            accuracy = self._num_corrects
        return accuracy