import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import front_end
from network import Conv2d
class Rnn(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_layer):
		super(Rnn, self).__init__()
		self.n_layer = n_layer
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
	def forward(self,x):
		out, _ = self.lstm(x)
		out = out[:,-1,:]
		return out

	