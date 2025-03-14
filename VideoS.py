import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
            frame_indices.append(idx)

        idx += 1

    cap.release()
    return frames, frame_indices

def extract_features(frames, batch_size=16):
    features = []
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    images = torch.stack([transform(img) for img in images]).to(device)   
    with torch.no_grad():  
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size] 
            feats = resnet(batch) 
            features.append(feats.cpu().numpy())  

    return np.concatenate(features, axis=0)

def select_keyframes(features, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    
    cluster_centers = kmeans.cluster_centers_
    keyframe_indices = []

    for center in cluster_centers:
        closest_idx = np.argmin(np.linalg.norm(features - center, axis=1))
        keyframe_indices.append(closest_idx)
    
    return keyframe_indices

def save_keyframes(frames, keyframe_indices, output_dir="keyframes"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, idx in enumerate(keyframe_indices):
        cv2.imwrite(f"{output_dir}/keyframe_{i}.jpg", frames[idx])
        print(f"Saved: keyframe_{i}.jpg")

def summarize_video(video_path, frame_interval=30, num_keyframes=5):
    frames, _ = extract_frames(video_path, frame_interval)
    features = extract_features(frames)
    keyframe_indices = select_keyframes(features, num_keyframes)
    save_keyframes(frames, keyframe_indices)


video_path = "sample.mp4" 
start = time.time()  
summarize_video(video_path, frame_interval=10, num_keyframes=15)
torch.cuda.synchronize()
end = time.time()
print(f"Generation of Keyframes on {device} took {end - start:.4f} seconds")