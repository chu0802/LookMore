from torchvision import transforms
import torch


def trans(examples, high_res_img_size=518, selector_img_size=154):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((high_res_img_size, high_res_img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]), 
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    ])
    examples["image"] = [transform(example) for example in examples["image"]]
    return examples

def trans_for_save(examples, img_size=518):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    examples["image"] = [transform(example) for example in examples["image"]]
    return examples