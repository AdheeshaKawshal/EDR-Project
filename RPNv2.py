import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
from torchvision.ops import nms

# ======= Dataset Definition =======
class SingleClassImageFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((68, 68)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('L')
        image = self.transform(image)

        label = torch.tensor(0.0 if 'gnd' in image_path.lower() else 1.0, dtype=torch.float)
        return image, label 

# ======= Model Definition =======
class SimpleCNNClassifier(nn.Module):
    def __init__(self):
        super(SimpleCNNClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# ======= Training Function =======
def train(model, device, dataset, epochs=5, batch_size=8, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'rpn2.pth')

# ======= Inference Function for Single Image =======
def infer_and_show_image(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((68, 68)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)

    input_batch = torch.stack([input_tensor.squeeze(0)] * 54).to(device)

    for _ in range(10):
        _ = model(input_batch)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model(input_batch)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Batch of 54 inference time: {(end - start) * 1000:.2f} ms")

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()

    plt.imshow(image)
    plt.title(f'Class {"1" if prob > 0.3 else "0"}: {prob:.4f}' if prob > 0.3 else f'Class 0: {1 - prob:.4f}')
    plt.axis('off')
    plt.show()

    return prob, end - start
objects1=[]
# ======= Sliding Window Inference =======
def sliding_window_batch_inference(model, image_path, device,
                                   window_size=100, stride=50,
                                   threshold=0.89, batch_size=64):
    model.eval()
    global objects1
    # orig_img = Image.open(image_path).convert('L')
    # orig_img_np = np.array(orig_img)
    # img_h, img_w = orig_img_np.shape
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        orig_img = frame
        orig_img_np = np.array(orig_img)
        img_h, img_w,_ = orig_img_np.shape
        if not ret:
            break

        # Convert frame for model input
        # Example: for classification
        # Display result on frame
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        transform = transforms.Compose([
            transforms.Resize((68, 68)),
            transforms.ToTensor()
        ])

        windows, coords = [], []
        for y in range(0, img_h - window_size + 1, stride):
            for x in range(0, img_w - window_size + 1, stride):
                patch = orig_img_np[y:y + window_size, x:x + window_size]
                tensor = transform(Image.fromarray(patch).convert('L'))

                windows.append(tensor)
                coords.append([x, y, x + window_size, y + window_size])
        if not windows:
            continue  # Skip this frame if no valid patches
        input_batch = torch.stack(windows).to(device)
        st = time.time()
        probs = []

        with torch.no_grad():
            for i in range(0, len(input_batch), batch_size):
                batch = input_batch[i:i + batch_size]
                out = model(batch)
                probs.extend(out.squeeze().cpu().numpy())

        print("Inference time:", time.time() - st, "patches:", len(windows))

        boxes, scores = [], []
        for i, prob in enumerate(probs):
            if prob > threshold:
                boxes.append(coords[i])
                scores.append(prob)

        if not boxes:
            print("No L-bends detected.")
            return []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores)
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.01)
        kept_boxes = boxes_tensor[keep].cpu().numpy().astype(int)
        kept_scores = list(scores_tensor[keep].cpu().numpy())
        #image_out = cv2.cvtColor(orig_img_np, cv2.COLOR_GRAY2BGR)
        image_out=orig_img_np
        i=0
        col=[(0,120,0),(0,0,240),(100,12,0),(0,120,180),(190,0,50),(0,120,250),(200,60,50)]
        sp=100
        lbendscore=0
        lind=0
        sorted_arr = np.sort(kept_scores)[::-1]
        lbendscore=np.argmax(kept_scores)
        for box in kept_boxes:
            x1, y1, x2, y2 = box
            val=kept_scores[i]
            if lbendscore<val:
                lbendscore=val
                lind=box
            label = f"Lbend {val}"
            # print(val)
            cv2.putText(image_out, label, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, .7, col[i % len(col)], 2, cv2.LINE_AA)
            cv2.rectangle(image_out, (x1, y1), (x2, y2), col[i%5], 2)
            if (x1,y1,x2,y2) not in objects1:objects1.append((x1,y1,x2,y2))
            if len(objects1)>20:objects1=objects1[10:]
            i+=1
            # cropped = orig_img_np[y1:y2+sp, x1:x2+sp]  # format: image[y1:y2, x1:x2]
            # cv2.imwrite("test2.jpg", cropped)
            # plt.figure(figsize=(10, 8))
            # plt.imshow(image_out)
            # plt.title(f"L-Bend Detect {image_path}")
            # plt.axis("off")
            # plt.show()
        # cv2.imwrite("test2.jpg", cropped)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(image_out)
        # plt.title(f"L-Bend Detect {image_path}")
        # plt.axis("off")
        # plt.show()
        cv2.imshow('Webcam Feed', image_out)

    cap.release()
    cv2.destroyAllWindows()



    return kept_boxes

# ======= Load Model and Run Detection =======
device = torch.device("cpu")
model = SimpleCNNClassifier().to(device)
model.load_state_dict(torch.load("rpn2.pth"))

image_path = "images\image_20250321_202824.jpg"
detected_boxes = sliding_window_batch_inference(model, image_path, device)
image_path1 = "test2.jpg"
# for i in range(2):
#     detected_boxes = sliding_window_batch_inference(model, image_path1, device)
