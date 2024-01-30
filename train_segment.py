from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import DataLoader
import wandb
from torchvision.transforms import functional


## ---------------
# test

import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # dla wszystkich GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

## ---------------



id2label = {
    0: "background",
    1: "blade"
}

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


class SegmentationDataset(Dataset):
    def __init__(self, root, target_dataset, split, transforms, device=None):
        self.target_root = os.path.join(root, target_dataset)
        self.support_root = root
        self.target_dataset = target_dataset
        self.split = split
        self.transforms = transforms
        self.device = device
        self.file_names = []
        self.images = []
        self.annotations = []
        self.image_dir = os.path.join(self.target_root, self.split)
        self.image_blade_dir = os.listdir(self.image_dir)
        self.image_dirs_all = [os.path.join(self.image_dir, x) for x in self.image_blade_dir]

        for file in self.image_dirs_all:
            if (file == './datasets/blade/train/Blade_4' or file == './datasets/blade/val/Blade_2'):  # smaller dataset  #if you want use full comment this line
                for blade in os.listdir(file):
                    if blade.endswith('jpg'):
                        self.images.append(os.path.join(file, blade))
                        self.annotations.append(os.path.join(file, 'mask', blade[:-4] + ".png"))


        assert len(self.images) == len(self.annotations)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        mask = Image.open(self.annotations[index]).convert('L')

        if self.transforms is not None:
            img, mask = self.apply_transform(img, mask)


        img = self.resize_img_to_shape(img, (280, 280))
        mask = self.resize_mask_to_shape(mask, (280, 280))
        mask_arr = mask.detach().cpu().numpy().squeeze(0)
        mask_arr[mask_arr < 0.5] = 0
        mask_arr[mask_arr >= 0.5] = 1
        mask = self.transforms(mask_arr)

        mask = mask.squeeze(0)
        img = img.squeeze(0)

        return img, mask.to(torch.int64)

    def __len__(self):
        return len(self.images)

    def resize_img_to_shape(self, img: torch.Tensor, desired_shape: tuple):
        img = torchvision.transforms.Resize((desired_shape[0], desired_shape[1]))(img)
        return img

    def resize_mask_to_shape(self, mask: torch.Tensor, desired_shape: tuple):
        mask = torchvision.transforms.Resize((desired_shape[0], desired_shape[1]))(mask)
        return mask

    def apply_transform(self,img,mask):
        if random.random()>0.5:
            angle=random.uniform(0,100)
            img=functional.rotate(img,angle)
            mask=functional.rotate(mask,angle)

        if random.random() > 0.5:
            img = functional.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))
            img = functional.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))
            img = functional.adjust_saturation(img, saturation_factor=random.uniform(0.5, 1.5))
            img = functional.adjust_hue(img, hue_factor=random.uniform(-0.1, 0.1))

        if random.random() > 0.5:
            img = functional.hflip(img)
            mask = functional.hflip(mask)

        if random.random() > 0.5:
            startpoints, endpoints = self.get_perspective_points(img.size)
            img = functional.perspective(img, startpoints, endpoints)
            mask = functional.perspective(mask, startpoints, endpoints)

        img = self.transforms(img)
        mask = self.transforms(mask)
        return img, mask

    def get_perspective_points(self, size):
        width, height = size
        # Define the degree of perspective change
        shift_max = min(width, height) * 0.2  # 20% of the smaller dimension
        # Define four points in the image from which to infer the perspective transform
        top_left = (random.uniform(-shift_max, shift_max), random.uniform(-shift_max, shift_max))
        top_right = (width - random.uniform(-shift_max, shift_max), random.uniform(-shift_max, shift_max))
        bottom_left = (random.uniform(-shift_max, shift_max), height - random.uniform(-shift_max, shift_max))
        bottom_right = (width - random.uniform(-shift_max, shift_max), height - random.uniform(-shift_max, shift_max))

        startpoints = [top_left, top_right, bottom_left, bottom_right]
        endpoints = [(0, 0), (width, 0), (0, height), (width, height)]

        return startpoints, endpoints


src_dataset_root = './datasets/'
train_dataset = SegmentationDataset(src_dataset_root, "blade","train", get_transform())
val_dataset = SegmentationDataset(src_dataset_root, "blade", "val", get_transform())

print("train_dataset shape: ", train_dataset.__len__())
print("val_dataset shape: ", val_dataset.__len__())


img, mask = train_dataset[3]
print(img.shape)
print(mask.shape)



def collate_fn(inputs):
    batch = dict()
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[0] for i in inputs]
    return batch


batchSize = 8
train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)


batch = next(iter(train_dataloader))
for k,v in batch.items():
    if isinstance(v,torch.Tensor):
        print(k,v.shape)

print(type(batch["original_images"]))
batch["labels"].dtype


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
       # self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
        # Dodanie dodatkowych warstw konwolucyjnych
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels // 2)  # Dodanie Batch Normalization
        self.act1 = torch.nn.LeakyReLU()  # Zmiana funkcji aktywacji na LeakyReLU

        self.conv2 = torch.nn.Conv2d(in_channels // 2, num_labels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_labels)

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        # Przechodzenie przez warstwy
        x = self.act1(self.bn1(self.conv1(embeddings)))
        x = self.bn2(self.conv2(x))

        return x
        # return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 20, 20, config.num_labels)
        #(image_width/14, image_height/14, it must be intergal number
        # 20,20 for (280,280)px
	    # 45,45 for (630,630) px

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
        outputs = self.dinov2(pixel_values,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label,num_labels=len(id2label))

# model_path='/home/student/ml/to_send/model_epoch_75.pth'
# model.load_state_dict(torch.load(model_path))



# print model layers
for name, param in model.named_parameters():
    # print in two columns
    print('{:<70} {}'.format(name, param.shape))

for name, param in model.named_parameters():
    if name.startswith("dinov2"):
        param.requires_grad = False



if isinstance(batch["original_images"], list):
    batch["original_images"] = torch.stack(batch["original_images"])

outputs = model(pixel_values=batch["original_images"], labels=batch["labels"])

print(outputs.logits.shape)
print(outputs.loss)


def apply_color_map(prediction):
    # Define colors for each class
    colors = {
        0: [255, 0, 0],  # Red for background
        1: [0, 255, 0]   # Green for blade
    }
    # Create a colored image based on class predictions
    colored_prediction = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_prediction[prediction == class_id] = color

    return colored_prediction

def tensor_to_pil(tensor):
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = TF.to_pil_image(tensor)
    return tensor


def overlay_predictions_on_image(image, prediction):
    # Apply color map to prediction
    colored_prediction = apply_color_map(prediction)

    # Convert colored prediction to PIL image
    colored_prediction_img = Image.fromarray(colored_prediction)

    # Overlay the colored prediction on the original image
    overlaid_image = Image.blend(image, colored_prediction_img, alpha=0.2)
    return overlaid_image


def save_image(image, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image.save(os.path.join(save_path, file_name))


def evaluate_model(model, val_dataloader, device, num_labels, save_path):
    model.eval()
    total_images = 0
    total_iou = 0.0
    f1_scores = []
    precisions = []
    recalls = []



    with torch.no_grad():
        print("\nValidation:")
        for idx, batch in enumerate(tqdm(val_dataloader)):  #tqdm(train_dataloader) val_dataloader
            pixel_values = torch.stack(batch["original_images"], dim=0).to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            logits = outputs.logits
            logits = F.interpolate(logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1)



            for i, prediction in enumerate(predictions):
                # overlaid_image = overlay_predictions_on_image(tensor_to_pil(batch['original_images'][i]),prediction.cpu().numpy())
                # save_image(overlaid_image, save_path, f"eval_image_{idx}_{i}.png")
                original_image_pil = tensor_to_pil(batch['original_images'][i])
                overlaid_image = overlay_predictions_on_image(original_image_pil, prediction.cpu().numpy())
                save_image(overlaid_image, save_path, f"eval_image_{idx}_{i}.png")

                # Obliczanie IoU dla klasy 1
                intersection = torch.logical_and(prediction == 1, labels[i] == 1)
                union = torch.logical_or(prediction == 1, labels[i] == 1)
                iou = torch.sum(intersection).item() / torch.sum(union).item() if torch.sum(union).item() > 0 else 0
                total_iou += iou
                total_images += 1

                # Obliczanie dla klasy 1
                true_positives = torch.sum((prediction == 1) & (labels[i] == 1)).item()
                predicted_positives = torch.sum(prediction == 1).item()
                actual_positives = torch.sum(labels[i] == 1).item()

                precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                recall = true_positives / actual_positives if actual_positives > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                f1_scores.append(f1_score)
                precisions.append(precision)
                recalls.append(recall)

    # Średnia IoU dla klasy 1
    average_iou = total_iou / total_images if total_images > 0 else 0

    # Średnie wartości dla klasy 1
    average_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0
    average_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    average_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0

    print(f'\nAverage Precision for class 1: {average_precision}')
    print(f'Average Recall for class 1: {average_recall}')
    print(f'Average F1 Score for class 1: {average_f1}')
    print(f'Average IoU for class 1: {average_iou}')

    wandb.log({"Average IoU": average_iou,
               "Average F1": average_f1,
               "Average Precision": average_precision,
               "Average Recall": average_recall})


    return average_iou, f1_scores, prediction, recalls
# training hyperparameters
# NOTE: I've just put some random ones here, not optimized at all
# feel free to experiment, see also DINOv2 paper


learning_rate = 0.0005
epochs = 50

run_number = 5

# # # ---------------------------------
wandb.init(project="blade_segmentation",entity='s176164')

# Konfiguracja hiperparametrów
config = wandb.config
config.learning_rate = learning_rate
config.epochs = epochs
config.batch_size = batchSize

# ---------------------------------

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for epoch in range(epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        pixel_values = torch.stack(batch["original_images"], dim=0).to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    #
    #
    wandb.log({"Train Loss": loss.item()})
    lr_scheduler.step()

    save_path = f'outputs/runs_{run_number}/saved_predictions_epoch_{epoch}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    evaluate_model(model, val_dataloader, device, 2, save_path)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'{save_path}/model_epoch_{epoch}.pth')

wandb.finish()
