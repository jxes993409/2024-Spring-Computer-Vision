import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
	def __init__(self):
		super(MyNet, self).__init__()
		self.conv1 = nn.Sequential(
		  nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
		  nn.ReLU(),
		  nn.MaxPool2d(kernel_size=2, stride=2),
			nn.BatchNorm2d(num_features=16),
		)
		self.conv2 = nn.Sequential(
		  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6)),
		  nn.ReLU(),
		  nn.MaxPool2d(kernel_size=2, stride=2),
			nn.BatchNorm2d(num_features=32),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(in_features=800, out_features=240),
			nn.ReLU(True),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(in_features=240, out_features=84),
			nn.ReLU(True),
		)
		self.fc3 = nn.Linear(in_features=84, out_features=10)
		self.drop_out = nn.Dropout2d(p=0.2)

	def forward(self, x):

		##########################################
		# TODO:                                  #
		# Define the forward path of your model. #
		##########################################

		x = self.conv1(x) # batch_size, 15, 15, 16
		x = self.conv2(x) # batch_size, 5, 5, 32
		x = torch.flatten(x, start_dim=1, end_dim=-1)
		x = self.fc1(x)
		x = self.drop_out(x)
		x = self.fc2(x)
		x = self.drop_out(x)
		x = self.fc3(x)

		return x
		pass

class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()

		############################################
		# NOTE:                                    #
		# Pretrain weights on ResNet18 is allowed. #
		############################################

		# (batch_size, 3, 32, 32)
		self.resnet = models.resnet18(pretrained=True)
		self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
		self.resnet.maxpool = Identity()
		# (batch_size, 512)
		self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
		# (batch_size, 10)

		#######################################################################
		# TODO (optinal):                                                     #
		# Some ideas to improve accuracy if you can't pass the strong         #
		# baseline:                                                           #
		#   1. reduce the kernel size, stride of the first convolution layer. # 
		#   2. remove the first maxpool layer (i.e. replace with Identity())  #
		# You can run model.py for resnet18's detail structure                #
		#######################################################################
				

	def forward(self, x):
		return self.resnet(x)

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

if __name__ == '__main__':
	model = ResNet18()
	# model = MyNet()
	# print(model)
