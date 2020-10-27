from datetime import datetime
from pytz import timezone
import time
import numpy as np
import torch
from collections import Counter


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return float(correct) * 100.0 / batch_size


def train(model, device, inputs, targets, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	i=0
	for input, target in iterate_minibatches(inputs,
												targets,
												batch_size=128,
												shuffle=True):
		# measure data loading time
		# if(len(input)!=32):
		# 	break

		input = torch.from_numpy(input)
		target = torch.from_numpy(target)
		input = input.view(-1, 1, 3000)
		target = target.long()
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)

		target = target.to(device)
		# print(target)
		optimizer.zero_grad()

		output = model(input)

		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target), target.size(0))
		if i % print_freq == 0:
			print('{0}  Epoch: [{1}][{2}/{3}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				datetime.now(timezone('EST')).strftime('%Y-%m-%d %H:%M:%S'),
				epoch, i*128, len(inputs), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))
			losses.reset()
			accuracy.reset()
		i=i+1
	return losses.avg, accuracy.avg


def evaluate(model, device, inputs, targets, criterion, print_freq=0):

	torch.cuda.manual_seed(42)

	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		i=0
		for input, target in iterate_minibatches(inputs,
												 targets,
												 batch_size=128,
												 shuffle=False):
			input = torch.from_numpy(input)
			target = torch.from_numpy(target)
			target = target.long()
			input = input.view(-1, 1, 3000)
			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))
			if (print_freq!=0):
				if i % print_freq == 0:
					print('Test: [{0}/{1}]\t'
						  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
						i*128, len(inputs), batch_time=batch_time, loss=losses, acc=accuracy))

				i=i+1
	return losses.avg, accuracy.avg, results
