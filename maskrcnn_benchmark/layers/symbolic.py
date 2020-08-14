import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.onnx.symbolic_opset9 import unsqueeze
from torch.onnx.symbolic_helper import parse_args

class NonMaxSuppression(Function):
    @staticmethod
    @parse_args('v', 'v', 'f', 'f', 'i')
    def symbolic(g, boxes, scores, iouThreshold, scoreThreshold=0.0, keepTopK=-1):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        if keepTopK == -1:
            keepTopK = boxes.size(0)
        iouThreshold = g.op('Constant', value_t=torch.tensor([iouThreshold], dtype=torch.float))
        scoreThreshold = g.op('Constant', value_t=torch.tensor([scoreThreshold], dtype=torch.float))
        keepTopK = g.op('Constant', value_t=torch.tensor([keepTopK], dtype=torch.int))
        return g.op("NonMaxSuppression", boxes, scores, iouThreshold, scoreThreshold, keepTopK)

    @staticmethod
    def forward(g, boxes, scores, iouThreshold, scoreThreshold=0.0, keepTopK=-1):
        if keepTopK == -1:
            keepTopK = boxes.size(0)
        return torch.ones(keepTopK, device=boxes.device, dtype=torch.long)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pass

