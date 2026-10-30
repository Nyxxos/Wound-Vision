import random
from pathlib import Path
import matplotlib.pyplot as plt
from gradcam import gradcam_heatmap
from dataset import data_path, selected_classes


for class_name in selected_classes:
    class_folder = Path(data_path) / class_name
    images = list(class_folder.glob("*.jpg"))
    images += list(class_folder.glob("*.jpeg"))
    images += list(class_folder.glob("*.png"))
    image_path = random.choice(images)
    heatmap, prediction, confidence, true_class, correct = gradcam_heatmap(image_path)
    
    print(f"\nImage: {image_path.name}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"True class: {true_class}")
    print(f"Correct: {correct}")
    
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title(f"True: {true_class}\nPredicted: {prediction} ({confidence:.2%})")
    plt.show()