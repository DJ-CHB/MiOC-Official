import os
import scipy.io as sio
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms



class TinyImagenet(Dataset):
      def __init__(self, root_dir, txt_file,  split='train', transform=None, dict_classes=None, moco=False):
          #get labels from .mat file
          self.root_dir = root_dir
          self.split = split
          self.transform = transform
          self.dict_classes = dict_classes
          self.moco = moco

          # Extract out the images with the bounding boxes
          self.path_images = []
          self.labels = []
          self.classes = os.listdir(self.root_dir)
          # Read the classes in .txt
          if self.split == 'train' or self.split == 'val':
              data_file = open((txt_file), "r")
              data_file = data_file.readlines()
              self.dict_classes = {}
              label = 0
              for classes in data_file:
                  #Remove the \n
                  classes = classes[:-1]
                  self.dict_classes[classes] = label
                  label += 1
                  path_classes = os.path.join(self.root_dir, classes, "images")
                  for images in os.listdir(path_classes):
                      self.path_images.append(os.path.join(path_classes, images))
                      self.labels.append(label)

          elif self.split == 'test':
                # Read classes and imagenames from val_annotations.txt
                data_file = open((txt_file), "r")
                data_file = data_file.readlines()
                #Print the first word
                for lines in data_file:
                    self.path_images.append(os.path.join(self.root_dir,  "images",  lines.split("\t")[0]))
                    self.labels.append(self.dict_classes[str(lines.split("\t")[1])])

      def __len__(self):
          return len(self.path_images)

      def __getitem__(self, index):
          image = Image.open(self.path_images[index])
          if self.split == 'train' or self.split=='val':
              label = self.dict_classes[self.path_images[index].split("/")[6]]
          elif self.split == 'test':
              label = self.labels[index]
          if image.mode == 'L':
              image = image.convert('RGB')

          if self.transform:
              image = self.transform(image)

          return image, label, "Empty"

#if __name__ == '__main__':
#  root_dir = "train"
#  txt_file = "wnids.txt"
#  transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
#  dataset = TinyImagenet(root_dir, txt_file,  split='train', transform=transform)
#  dict_classes = dataset.dict_classes
#  txt_file_test = "test/val_annotations.txt"
#  root_dir = "test"
#  dataset_test = TinyImagenet(root_dir, txt_file_test,  split='test', transform=transform, dict_classes=dict_classes)
#  print(dataset[0][1])
#  print(len(dataset_test))
#  print(dataset_test[0][0].shape)
