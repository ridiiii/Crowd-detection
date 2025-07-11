#  Crowd Detection Using YOLOv8 and OpenCV

This project implements an intelligent *Crowd Detection System* using a custom-trained **YOLOv8 model**. It identifies and tracks groups of people (crowds) in video footage based on spatial proximity and temporal persistence.


##  Features

- ✅ Real-time person detection using YOLOv8
- ✅ Crowd detection based on spatial closeness and group size
- ✅ Crowd persistence tracking over consecutive frames
- ✅ Annotated output video with visual crowd indicators
- ✅ CSV report logging all detected crowd events


##  Objective

To detect **crowds** in a video stream by identifying groups of **three or more people** who are **standing close (within 100 pixels)** and **persist for at least 10 frames**.


##  How It Works

1. **Person Detection**: Each frame is passed through a custom YOLOv8 model (`crowd.pt`) to detect persons.
2. **Center Calculation**: Bounding box centers of detected persons are computed.
3. **Group Formation**: Individuals within a defined Euclidean distance are grouped.
4. **Crowd Logic**: If a group has ≥ 3 people and exists for ≥ 10 frames, it's flagged as a crowd.
5. **Output**:
   - Annotated video with red circles and crowd labels
   - CSV file logging frame numbers and crowd sizes





