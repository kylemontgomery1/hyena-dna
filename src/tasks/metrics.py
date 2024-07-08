import torch
import torch.nn.functional as F
import torchmetrics.functional as tm_f

def cross_entropy(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    loss = F.cross_entropy(logits, y, ignore_index=ignore_index)
    return loss

def accuracy_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    return tm_f.classification.binary_accuracy(preds, y, ignore_index=ignore_index)

def f1_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    return tm_f.classification.binary_f1_score(preds, y, ignore_index=ignore_index)
    
def precision_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    return tm_f.classification.binary_precision(preds, y, ignore_index=ignore_index)

def recall_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    return tm_f.classification.binary_recall(preds, y, ignore_index=ignore_index)

def confusion_matrix_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    return tm_f.classification.binary_confusion_matrix(preds, y, ignore_index=ignore_index)

def auroc_binary_ignore_index(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    probs = F.softmax(logits, dim=-1)
    preds = probs[..., 1]
    y = y.view(-1)
    return tm_f.classification.binary_auroc(preds, y, ignore_index=ignore_index)

# should have a better way to do this
metric_fns = {
    "cross_entropy": cross_entropy,
    "accuracy_binary_ignore_index": accuracy_binary_ignore_index,
    "f1_binary_ignore_index": f1_binary_ignore_index,
    "precision_binary_ignore_index": precision_binary_ignore_index,
    "recall_binary_ignore_index": recall_binary_ignore_index,
    "confusion_matrix_binary_ignore_index": confusion_matrix_binary_ignore_index,
    "auroc_binary_ignore_index": auroc_binary_ignore_index,
}