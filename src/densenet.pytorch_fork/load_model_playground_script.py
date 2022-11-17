import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def get_transform(dataset_name=""):
    normalize_transform = transforms.Normalize( # default is CIFAR-10
        (125.3/255, 123.0/255, 113.9/255),
        (63.0/255, 62.1/255.0, 66.7/255.0)
    )
    if dataset_name == "SVHN":
        normalize_transform = transforms.Normalize(
            (0.43768218, 0.44376934, 0.47280428), 
            (0.1980301, 0.2010157, 0.19703591)
        )
    return transforms.Compose([
        transforms.ToTensor(),
        normalize_transform
    ])

net = torch.load("./Densenet_Train_SVHN_4.3_Percent_Error.pth", map_location=torch.device('cpu'))

testset_out = torchvision.datasets.SVHN(root='svhn', split='test', download=True, transform=get_transform("SVHN"))
testloader_out = torch.utils.data.DataLoader(testset_out, batch_size=1,shuffle=False)

for j, data in enumerate(testloader_out):
    images, _ = data

    inputs = Variable(images, requires_grad = True)
    outputs = net(inputs)
    outputs = outputs.detach().numpy()
    nnOutputs = np.exp(outputs)/np.sum(np.exp(outputs))
    print("nnOutputs",nnOutputs)
    plt.imshow(  inputs[0].permute(1, 2, 0).detach().numpy()  )
    break

plt.show()