from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import torch

class CrowdDetector:
    def __init__(self, model_path, distance_threshold=100, crowd_threshold=3, frame_threshold=10):
        
        self.model = YOLO(model_path)
        self.distance_threshold = distance_threshold
        self.crowd_threshold = crowd_threshold
        self.frame_threshold = frame_threshold
        self.crowd_history = defaultdict(int)
        self.results = []

    def calculate_center(self, box):
        
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def get_groups(self, centers):
        
        groups = []
        used = set()

        for i, center1 in enumerate(centers):
            if i in used:
                continue

            current_group = {i}
            for j, center2 in enumerate(centers):
                if j in used:
                    continue
                    
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 + 
                    (center1[1] - center2[1])**2
                )
                
                if distance < self.distance_threshold:
                    current_group.add(j)
                    used.add(j)

            if len(current_group) >= self.crowd_threshold:
                groups.append(current_group)

        return groups

    def process_frame(self, frame, frame_number):
        """Process a single frame and detect crowds"""
        
        results = self.model(frame, verbose=False)[0]
        
        
        persons = []
        for box in results.boxes:
            if box.cls == 0:  # person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                persons.append([x1, y1, x2, y2])
        
        persons = np.array(persons)
        
        if len(persons) < self.crowd_threshold:
            return frame, []

       
        centers = [self.calculate_center(box) for box in persons]
        
      
        groups = self.get_groups(centers)
        
       
        crowds = []
        for group_idx, group in enumerate(groups):
            group_id = f"crowd_{group_idx}"
            self.crowd_history[group_id] += 1
            
            
            if self.crowd_history[group_id] >= self.frame_threshold:
                crowds.append({
                    'frame_number': frame_number,
                    'person_count': len(group)
                })
                
                
                center = np.mean([centers[i] for i in group], axis=0)
                cv2.circle(frame, (int(center[0]), int(center[1])), 
                          50, (0, 0, 255), 2)
                cv2.putText(frame, f'Crowd: {len(group)} people', 
                          (int(center[0]), int(center[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, crowds

    def process_video(self, video_path, output_path='output.mp4', csv_path='crowds.csv'):
        "Process entire video and save results"
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            
            processed_frame, crowds = self.process_frame(frame, frame_number)
            
            # Save results
            self.results.extend(crowds)
            
            
            out.write(processed_frame)
            
            frame_number += 1
            
            
            if frame_number % 30 == 0:  
                print(f"Processed frame {frame_number}")
       
        cap.release()
        out.release()
        
        # Save results to CSV
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

def main():
  
    detector = CrowdDetector(
        model_path=r"crowd.pt",
        distance_threshold=100,  
        crowd_threshold=3,
        frame_threshold=10
    )
    
    # Process video 
    detector.process_video(
        video_path='dataset_video.mp4',
        output_path='output_video.mp4',
        csv_path='crowd_detections.csv'
    )

if __name__ == "__main__":
    main()
