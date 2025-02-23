from __future__ import division
import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import gradio as gr
from numpy.core.multiarray import scalar
import bisect
from scipy.stats import norm
from your_model_file import HRNet_dropout
from matplotlib.cm import get_cmap
import Utility_functions


device = torch.device("cpu")
n_MC = 10
thresh = 0.75
CL = 95

# Configuration settings
hypes = {
    "arch": {
        "ALIGN_CORNERS": False,
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "BLOCK": "BOTTLENECK",
                "FUSE_METHOD": "SUM",
                "NUM_BLOCKS": [4],
                "NUM_BRANCHES": 1,
                "NUM_CHANNELS": [64],
                "NUM_MODULES": 1
            },
            "STAGE2": {
                "BLOCK": "BASIC",
                "FUSE_METHOD": "SUM",
                "NUM_BLOCKS": [4, 4],
                "NUM_BRANCHES": 2,
                "NUM_CHANNELS": [48, 96],
                "NUM_MODULES": 1
            },
            "STAGE3": {
                "BLOCK": "BASIC",
                "FUSE_METHOD": "SUM",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_BRANCHES": 3,
                "NUM_CHANNELS": [48, 96, 192],
                "NUM_MODULES": 4
            },
            "STAGE4": {
                "BLOCK": "BASIC",
                "FUSE_METHOD": "SUM",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_BRANCHES": 4,
                "NUM_CHANNELS": [48, 96, 192, 384],
                "NUM_MODULES": 3
            }
        },
        "bayes": False,
        "config": "HRNet_do",
        "image_shape": [3, 512, 512],
        "num_classes": 1,
        "recon": False
    },
    "data": {
        "background_colour": [255, 0, 0],
        "class_colours": [[255, 0, 0], [255, 0, 255]],
        "class_labels": ["background", "corrosion"],
        "class_weights": [1.0, 7.035],
        "overlay_colours": [[0, 255, 0, 0], [255, 0, 255, 200]],
        "pop_mean": [0.5596, 0.4997, 0.4767],
        "pop_std0": [0.2066, 0.2224, 0.2382]
    },
    "model": "fold8_epoch100.pth"}

input_res = hypes['arch']['image_shape'][1:3]
input_transforms = transforms.Compose([transforms.Resize(input_res),transforms.ToTensor(),
    transforms.Normalize(hypes['data']['pop_mean'], hypes['data']['pop_std0'])])

# ------------ Classes for Grad-CAM
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"], self.model(x)['logVar']

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask).float()
        self.mask = self.mask.to('cpu')

    def __call__(self, model_output):
      print('Segment output', (model_output[:, :, :] * self.mask).sum())
      return (torch.relu(model_output[:, :, :]) * self.mask).sum()
        
# --------- load model and assign trained weights
seg_model = HRNet_dropout(config=hypes).to(device)
torch.serialization.add_safe_globals([scalar])
model_path = os.path.join(os.path.dirname(__file__), hypes['model'])
with torch.serialization.safe_globals([scalar]):
    pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)
if 'state_dict' in pretrained_dict:
    pretrained_dict = pretrained_dict['state_dict']
prefix = "module."
for key in list(pretrained_dict.keys()):
    if key.startswith(prefix):
        newkey = key[len(prefix):]
        pretrained_dict[newkey] = pretrained_dict.pop(key)

seg_model.load_state_dict(pretrained_dict)
seg_model.eval()

# ----------- prediction function
def corrosion_detection(files):
    
    all_cam_images = []
    all_classes = []
    all_percentages = []
    all_overlay_images = []
    total_images = len(files)

    # ---------- Initial values
    yield (gr.update(value=f"Processing image {1}/{total_images}... Please wait."),
           gr.update(value=all_cam_images), 
           gr.update(value="\n".join(all_classes)), 
           gr.update(value="\n".join(all_percentages)), 
           gr.update(value=all_overlay_images))
    
    for idx, file in enumerate(files):
        image = Image.open(file)
        image_orig = image
        image_array = np.array(image_orig) / 255.0
        input_tensor = input_transforms(image_orig).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        model = SegmentationModelOutputWrapper(seg_model)
        model.train()
        with torch.no_grad():
            input_tensor_batch = input_tensor.repeat(n_MC, 1, 1, 1)
            outputs = model(input_tensor_batch)
            preds, logVars = outputs
            preds = preds.squeeze(1)
            logVars = logVars.squeeze(1)
        
        mean_prediction_init = preds.mean(dim=0)
        logvar_prediction = logVars.mean(dim=0)
        
        # --------- Create the Mask
        min_value, max_value = mean_prediction_init.min(), mean_prediction_init.max()
        mean_prediction = (mean_prediction_init - min_value) / (max_value - min_value)
        mask = (mean_prediction >= thresh).float()

        # --------- Grad-CAM for the Mean Corroded Area
        target_layer = [seg_model.last_layer[3]]
        mask = np.float32(mask)
        targets = [SemanticSegmentationTarget(0, mask)]
        image_array = cv2.resize(np.array(image_array), (input_res[0], input_res[1]))
        with GradCAM(model=model, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)

        # --------- Uncertainty Estimation
        epistemic_variance = preds.var(dim=0)
        aleatoric_variance = torch.exp(logVars).mean(dim=0)
        variance = epistemic_variance + aleatoric_variance
        z_score = norm.ppf(1 - (1 - CL / 100) / 2)
        std_dev = torch.sqrt(variance) / np.sqrt(n_MC)
        upper_prediction = mean_prediction_init + z_score * std_dev
        upper_prediction = torch.clamp(upper_prediction, 0, 1)
        upper_mask = (upper_prediction >= thresh).float()

        # --------- Corrosion Severity Calculation
        mask = torch.tensor(mask, device=device)
        mean_score = round(torch.sum(mask == 1).item() / mask.numel() * 100, 2)
        upper_score = round(torch.sum(upper_mask == 1).item() / upper_mask.numel() * 100, 2)
        thresholds = [5, 10, 15, 25, 35, 45, 55, 65, 75, 85]
        classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5',
                   'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']
        cor_class = classes[bisect.bisect_right(thresholds, mean_score)]

        # --------- Create image of corroded area
        mask_resized = cv2.resize(mask.cpu().numpy(), (input_res[0], input_res[1]))
        mask_colored = np.zeros((input_res[1], input_res[0], 4), dtype=np.float32)
        mask_colored[..., :3] = (139 / 255, 69 / 255, 19 / 255) 
        mask_colored[..., 3] = mask_resized * 0.7

        image_orig_resized = Image.fromarray((image_array * 255).astype(np.uint8))
        overlay_image = Image.alpha_composite(image_orig_resized.convert("RGBA"), Image.fromarray((mask_colored * 255).astype(np.uint8), "RGBA"))
        
        # --------- Accumulate Outputs
        if cam_image is not None:
            all_cam_images.append((cam_image, "CAM Image"))
        if overlay_image is not None:
            all_overlay_images.append((overlay_image, "Overlay Image"))
        if cor_class:
            all_classes.append(f"Image {idx+1}: {cor_class}")
        if mean_score:
            all_percentages.append(f"Image {idx+1}: {mean_score}% ± {round(-upper_score + mean_score, 2)}%")
        
        # --------- Update Status Bar
        status_message = f"Processing image {idx+2}/{total_images}... Please wait."
        yield (gr.update(value=status_message), gr.update(value=all_cam_images), 
               gr.update(value="\n".join(all_classes)), 
               gr.update(value="\n".join(all_percentages)), 
               gr.update(value=all_overlay_images))
    
    # --------- Final Completion Message
    yield (gr.update(value="✅ Processing complete! All images have been analysed."),
           gr.update(value=all_cam_images), 
           gr.update(value="\n".join(all_classes)), 
           gr.update(value="\n".join(all_percentages)), 
           gr.update(value=all_overlay_images))

# ------------ create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("### Corrosion Detection with Grad-CAM")

    image_input = gr.Files(label="Upload Images", file_types=[".png", ".jpg", ".jpeg"])
    process_button = gr.Button("Submit Images")

    gr.Markdown("### Detection Results:")

    status_box = gr.Textbox(label="Processing Status", interactive=False)
    overlay_gallery = gr.Gallery(label="Corrosion Overlay", columns=4)

    gr.Markdown("### Grad-CAM Explanation: Explains Corroded Segments Using Gradients")
    output_gallery = gr.Gallery(label="Grad-CAM Heatmap", columns=4)
    corrosion_classes = gr.Textbox(label="Corrosion Classes")
    corroded_percentages = gr.Textbox(label="Corroded Percentages")

    process_button.click(corrosion_detection, inputs=image_input, 
                         outputs=[status_box, output_gallery, corrosion_classes, corroded_percentages, overlay_gallery])

iface.launch()

