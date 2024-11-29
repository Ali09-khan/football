from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from utils import get_bbox_width, get_center_of_bbox, get_foot_position

class Tracker:
  def __init__(self, ball_model_path, players_model_path):
    self.ball_model = YOLO(ball_model_path)
    self.players_model = YOLO(players_model_path)
    self.byte_tracker = sv.ByteTrack()

  def detect_frames(self, frames):
    batch_size = 20
    players_detections = []
    ball_detections = []
    for i in range(0, len(frames), batch_size):
      batch_detec_play = self.players_model.predict(frames[i:i+batch_size])
      batch_detec_ball = self.ball_model.predict(frames[i:i+batch_size])
      players_detections += batch_detec_play
      ball_detections += batch_detec_ball
    return players_detections, ball_detections

  def get_object_tracks(self, player_detections, ball_detections):
    tracks = {
        "players" : [],
        "referees" : [],
        "goalkeepers" : [],
        "ball" : []
    }
    if len(ball_detections) != len(player_detections):
      raise ValueError("The length of ball_detection and player_detection are not the same")

    for i in range(0, len(player_detections)):
      players_names = player_detections[i].names
      players_names_inv = {v:k for k,v in players_names.items()}

      players_detection_supervision = sv.Detections.from_ultralytics(player_detections[i])
      ball_detection_supervision = sv.Detections.from_ultralytics(ball_detections[i])
      players_detection_wtids = self.byte_tracker.update_with_detections(players_detection_supervision)

      tracks["players"].append({})
      tracks["referees"].append({})
      tracks["goalkeepers"].append({})
      tracks["ball"].append({})

      for player_detection in players_detection_wtids:
        bbox = player_detection[0].tolist()
        cls_id = player_detection[3]
        track_id = player_detection[4]

        if cls_id == players_names_inv["player"]:
          tracks["players"][i][track_id] = {"bbox":bbox}

        if cls_id == players_names_inv["referee"]:
          tracks["referees"][i][track_id] = {"bbox":bbox}

        if cls_id == players_names_inv["goalkeeper"]:
          tracks["goalkeepers"][i][track_id] = {"bbox":bbox}

      tracks['ball'][i][1] = {"bbox" : np.array(ball_detection_supervision.xyxy.ravel()).tolist()}#np.array(ball_detection_supervision.xyxy.ravel()).tolist()

    return tracks

  def draw_ellipse(self, frame, bbox, color):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color = color,
        thickness=2,
        lineType=cv2.LINE_4
    )
    return frame

  def draw_triangle(self, frame, bbox, color):
    if len(bbox) != 0:
      y = int(bbox[1])
      x,_ = get_center_of_bbox(bbox)
      triangle_points = np.array([
          [x,y],
          [x-10,y-20],
          [x+10,y-20],
      ])
      cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
      cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)
    return frame

  def draw_annotations(self, video_frames, tracks):
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()
      player_dict = tracks["players"][frame_num]
      ball_dict = tracks["ball"][frame_num]
      referee_dict = tracks["referees"][frame_num]
      goalkeeper_dict = tracks["goalkeepers"][frame_num]

      for _, player in player_dict.items():
        color = player.get("team_color",(0,0,255))
        frame = self.draw_ellipse(frame, player["bbox"],color)
      for _, referee in referee_dict.items():
        frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
      for _, goalkeeper in goalkeeper_dict.items():
        frame = self.draw_ellipse(frame, goalkeeper["bbox"],(255,255,255))
      #ball
      for _, ball in ball_dict.items():
        frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))
      output_video_frames.append(frame)
    return output_video_frames