# Corrosion-Detection-GradCAM
This project detects corrosion in images using a pre-trained CNN model and visualizes results. The system provides an image marking the corroded area, Grad-CAM explanations, corrosion percentage estimation, and classification based on the corroded percentage.

The developed web application from this approach is available at: https://huggingface.co/spaces/Shashikatd/Corrosion_Detector

## How the Application Works

This application processes images of steel members to detect and segment areas of corrosion. It assigns a corrosion severity rating on a scale of **Class 1 to Class 10** based on the detected corrosion percentage. furthermore, it provides **Grad-CAM visualizations** to explain the model’s predictions.

### Workflow:
1️⃣ **User uploads images** of steel members.  
2️⃣ **Model processes each image**, segmenting corrosion areas.  
3️⃣ **Grad-CAM visualization is generated** to explain the model’s focus.  
4️⃣ **Corrosion severity (%) is calculated** and mapped to a **corrosion class (1-10)**.  
5️⃣ **Outputs are displayed per image**, including:
   - **Corrosion overlay image** (visual segmentation of corroded regions).  
   - **Grad-CAM explanation** (heatmap visualization for model interpretability).  
   - **Corrosion class** (severity rating from 1 to 10).  
   - **Corroded percentage** (quantification of corrosion).  

### Features:
- **Batch processing** of multiple images.  
- **Real-time updates** with a status indicator.  
- **Explainable AI (XAI)** through Grad-CAM.  
- **Dynamic output visualization** for each image.  

### 🔹 **Figure 1: Processing Status**
- The detection of corroded segments for the **first image** is completed.
- Results are visualized while other images are still being processed.
- The **status bar** provides guidance on the analysis progress.

![Processing Image 1](https://github.com/janavodnirmalj/Corrosion-Detection-GradCAM/blob/main/WebApp_Processing.png)

---

### 🔹 **Image 2: Final Output for Multiple Image Submission**
- This figure showcases the **final output** when all images have been processed.
- Each image receives an **individual corrosion analysis**.
- The interface displays:
  - Corrosion overlay image.
  - Grad-CAM explanation.
  - Corrosion class (Severity from 1 to 10).
  - Corroded percentage.

![Final Output](https://github.com/janavodnirmalj/Corrosion-Detection-GradCAM/blob/main/WebApp_Output.png)

## Sample Output
A sample output for a given input image is summarised in the following table:

| Input Image | Grad-CAM Heatmap | Corrosion Mask | Corrosion Class | Corroded Percentage |
|-------------|-----------------|----------------|-----------------|-------------------|
| ![Input](https://github.com/janavodnirmalj/Corrosion-Detection-GradCAM/blob/main/Image1.png)| ![Grad-CAM](https://github.com/janavodnirmalj/Corrosion-Detection-GradCAM/blob/main/Image1_GC.png) | ![Mask](https://github.com/janavodnirmalj/Corrosion-Detection-GradCAM/blob/main/Image1_OL.png) | Class 5 | 30.53% ± 14.49% |

