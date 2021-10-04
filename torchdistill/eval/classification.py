import torch


def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])
        result_list = []
        for k in topk:
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            result_list.append(correct_k * (100.0 / batch_size))
        return result_list

def masked_index(input, dim, mask):
	assert len(mask.size()) == 1 and input.size(dim) == mask.size(0), \
		'{}!=1 or {}!={}'.format(len(mask.size()), input.size(dim), mask.size(0))
	indices = torch.nonzero(mask)
	return torch.index_select(input, dim, indices.squeeze())

def masked_compute_accuracy(outputs, targets, target_mask, num_counts, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    """Except for masks """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None]).float()
        
        # remove invalid target class cases
        corrects = masked_index(corrects, 1, target_mask)
        
        result_list = []
        for k in topk:
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            result_list.append(correct_k * (100.0 / num_counts))
        return result_list
