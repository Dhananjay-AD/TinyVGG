from torch.utils.data import Dataset
import torchvision
from pathlib import Path
from PIL import Image

class CustomImageDataset(Dataset):
    """
    Class to get custom dataset in the format torch.utils.data.Dataset
    """
    def __init__(self,
                dataset_path,
                transform: torchvision.transforms
                ):
        self.dataset_path = dataset_path
        self.transform = transform
        self.img_paths = self.path_to_jpgpaths()
        self.class_to_idx, self.classes = self.classes_and_dict()

    def __getitem__(self, index):
        target_path = self.img_paths[index]
        imgs_pil = self.load_image(target_path)
        img_transformed = self.transform(imgs_pil)
        return (img_transformed, self.class_to_idx[target_path.parent.stem])

    def __len__(self):
        return len(self.img_paths)

    def path_to_jpgpaths(self):
        img_paths = sorted(self.dataset_path.glob('*/*.jpg'))
        return img_paths

    def classes_and_dict(self):
        classes = [x.stem for x in sorted(self.dataset_path.glob('*'))]
        # getting dictionary
        classes_dict = {label: index for index, label in enumerate(classes)}
        return classes_dict, classes

    def load_image(self, img_paths):
        imgs_pil = Image.open(img_paths)
        return imgs_pil
