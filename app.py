import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

MODEL_PATH = "./monkey_resnet18.pth"

CLASS_NAMES = [
    "Mantled Howler",
    "Patas Monkey",
    "Bald Uakari",
    "Japanese Macaque",
    "Pygmy Marmoset",
    "White Headed Capuchin",
    "Silvery Marmoset",
    "Common Squirrel Monkey",
    "Black Headed Night Monkey",
    "Nilgiri Langur"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# Prediction Function
# =====================================================

def predict(image: Image.Image):

    image = image.convert("RGB")

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return results

# =====================================================
# Gradio UI
# =====================================================

description = """
Upload a monkey image and the model will predict the species.

Model:
- ResNet18 Transfer Learning
- PyTorch
- 10 Monkey Species Dataset
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🐒 Monkey Species Classifier",
    description=description,
    allow_flagging="never"
)

# =====================================================
# Launch
# =====================================================

if __name__ == "__main__":
    demo.launch()