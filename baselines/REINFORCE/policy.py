import torch
import torch.nn as nn

class policy_network(nn.modules):
	def __init__(self, in_dim, hidden, num_action):
		self.model = nn.Sequential(
					nn.Linear(in_dim, hidden),
					nn.tanh(),
					nn.Linear(hidden, num_action))
		self.softmax = nn.Softmax(dim = -1)

	def forward(self, x):
		out = self.model(x)	
		out = self.softmax(out)
		return out
		
