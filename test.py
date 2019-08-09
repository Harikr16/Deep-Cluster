import torch
from torchvision import models
import time

data = torch.randn(8,3,512,512).cuda()
targets = torch.randint(0,5,(8,)).cuda()

data_h = data.half()

model = models.densenet201().cuda()
model_h = models.densenet201().half().cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
optimizer_h = torch.optim.SGD(model_h.parameters(),lr = 0.01)

start = time.time()
for i in range(20):
	output = model(data)
	loss = criterion(output,targets)
	loss.backward()
	optimizer.step()
end = time.time()
print("Time taken for 10 iterations of float type = ", end - start)

start = time.time()
for i in range(20):
	output = model_h(data_h)
	loss = criterion(output,targets)
	loss.backward()
	optimizer_h.step()
end = time.time()
print("Time taken for 10 iterations of half type = ", end - start)
