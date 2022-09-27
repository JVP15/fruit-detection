import numpy as np
import tensorflow as tf

from typing import List, Tuple

classnames = ['unhealthy', 'healthy']

class DiseaseModule(object):
    def __init__(self, weights_path, device='/GPU:0'):
        self.device = device


        with tf.device(device):
            self.model = tf.keras.models.load_model(weights_path)

    def get_disease_predictions(self, img_batch: tf.Tensor) -> List[Tuple[int, float]]:
        # if there are no images in the input batch, we just return an empty list
        if img_batch.shape[0] > 0:
            with tf.device(self.device):
                predictions = self.model(img_batch)

            disease_predictions = []
            for prediction in predictions:
                disease_predictions.append((np.argmax(prediction), np.max(prediction))) #TODO: do this using vectorized functions

            return disease_predictions
        else:
            return []

def eval(net, loader, type):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(loader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    sys.stdout.flush()
    print(f'Accuracy of the network on the {total} images from {type} dataset: {100 * correct // total}%')


if __name__ == '__main__':
    import torch.optim as optim
    import torch.nn as nn
    import torch
    import torchvision
    from torchvision import transforms
    from fruit_cnn import FruitCNN
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import sys

    train_folder = 'apple_defect_detection\\apple_dataset_normal_and_defected\\apple_dataset_normal_and_defected\\train'
    test_folder = 'apple_defect_detection\\apple_dataset_normal_and_defected\\apple_dataset_normal_and_defected\\validation'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(train_folder, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(test_folder, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # get some images and labels from the training dataloader and display them using Matplotlib
    for images, labels in train_loader:
        for i in range(8):
            plt.imshow(images[i].numpy().transpose(1, 2, 0))
            plt.title(classnames[labels[i]])
            plt.show()

        break

    net = FruitCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(100):  # loop over the dataset multiple times
        print('Epoch:', epoch)
        net.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Steps'):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        eval(net, train_loader, 'train')
        eval(net, test_loader, 'test')


    print('Finished Training')

