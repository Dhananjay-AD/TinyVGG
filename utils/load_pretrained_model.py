from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights
from torchinfo import summary
import torch

from utils.custom_dataset import CustomImageDataset
from pathlib import Path

def load_pretrained_model():
    """
    Loads the efficientnet_b0 model, freeze the core model layers to refrain from changing pretrained parameters
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    pre_trained_model = efficientnet_b0(weights = weights)
    transform = weights.transforms()
    #print(transform)
    
    # freezing the trainable params of feature part
    for param in pre_trained_model.features.parameters():
        param.requires_grad = False

    training_path = Path.cwd().parent/'dataset'/'custom_dataset'/'testing_dataset'

    # getting available class information
    custom_dataset = CustomImageDataset(dataset_path = training_path,
                                                 transform = transform)
    
    print(len(custom_dataset.classes))
    # changed the classifier part of the model in such a way that it will match the number of classes we have
    pre_trained_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features = 1280,
                        out_features = 3)
    )

    # summary
    summary(model = pre_trained_model,
        input_size = (32, 3, 224, 224),
        col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
        col_width = 20,
        row_settings=["var_names"])

    return pre_trained_model, transform


if __name__ == '__main__':
    load_pretrained_model()