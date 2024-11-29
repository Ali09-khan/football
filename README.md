# Football Analysis

This repository provides an AI-powered platform for analyzing football match broadcast videos. The pipeline incorporates state-of-the-art deep learning techniques for **field localization**, **player detection and tracking**, **keypoint detection**, **ball tracking**, and **homography estimation**.

---

## **Pipeline Overview**

1. **Field Localization**:
   - Uses the **SegFormer** model (a Transformer-based semantic segmentation framework) for precise identification of soccer field markings and boundaries.

2. **Player and Keypoint Detection**:
   - **Model**: YOLO11l (Large version of YOLO version 11).
   - **Task**: Detect players and their keypoints for pose estimation.
   - **Note**: This provides a foundation for analyzing player movements and actions.

3. **Ball Detection**:
   - **Model**: YOLO11s (Small version of YOLO version 11).
   - **Task**: Detect and track the ball in each frame for trajectory analysis.

4. **Player Tracking**:
   - **Algorithm**: BoT-SORT tracker is employed to maintain robust player identities throughout the video.

5. **Homography Estimation**:
   - **Model**: HomographyNet, a convolutional neural network for directly estimating homographies between image pairs.
   - **Task**: Align frames with a reference perspective for consistent spatial analysis.

---

## **Key Features**

- **Field Localization**: High accuracy segmentation of the field using SegFormer.
- **Player Detection**: Precise detection of players and keypoints with YOLO11l.
- **Ball Detection**: Lightweight YOLO11s model optimized for tracking the ball.
- **Homography Alignment**: Robust alignment of broadcast video frames with a predefined field template.
- **Player Tracking**: Seamless tracking of player movements throughout the match.
