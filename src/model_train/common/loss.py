import torch

def calSmoothCrossEntropy(predictions, golds):
    loss = 0.0
    for i, scores in enumerate(predictions):
        scores = scores + 1e-7
        loss += golds[i][0] * torch.log(scores[0]) + golds[i][1] * torch.log(scores[1])
        # import pdb; pdb.set_trace()
    loss = 0 - loss
    return loss


def calReluSmoothCrossEntropy(predictions, golds, smooth):
    loss = 0.0
    for i, scores in enumerate(predictions):
        scores = scores + 1e-7
        if(i == 0):
            if(scores[1] <= (1 - smooth)):
                loss += torch.log(scores[1])
        else:
            if(scores[0] >= smooth):
                loss += torch.log(scores[0])
        # loss += golds[i][0] * torch.log(scores[0]) + golds[i][1] * torch.log(scores[1])
        # import pdb; pdb.set_trace()
    loss = 0 - loss
    return loss