from configs import SoccerPitchConfiguration
import supervision as sv
import numpy as np
from ViewTransformer import ViewTransformer
from utils import return_pitch_xy, team_comparison
from Annotators import draw_pitch, draw_points_on_pitch

CONFIG = SoccerPitchConfiguration()
class MapMaker:
  def __init__(self, model):
    self.model = model

  def create_map(self, video_frames, tracks):
    i=0
    annotated_frames = []
    for frame in video_frames:
      result = self.model.predict(frame, conf=0.3)[0]
      key_points = sv.KeyPoints.from_ultralytics(result)

      filter = key_points.confidence[0] > 0.5
      frame_reference_points = key_points.xy[0][filter]
      pitch_reference_points = np.array(CONFIG.vertices)[filter]

      transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

      frame_ball_xy = return_pitch_xy(tracks, "ball", i)
      players_xy = return_pitch_xy(tracks, "players", i)
      referees_xy = return_pitch_xy(tracks, "referees", i)
      goalkeepers_xy = return_pitch_xy(tracks, "goalkeepers", i)

      pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
      pitch_players_xy = transformer.transform_points(points=players_xy)
      pitch_referees_xy = transformer.transform_points(points=referees_xy)
      pitch_goalkeepers_xy = transformer.transform_points(points=goalkeepers_xy)

      annotated_frame_2 = draw_pitch(CONFIG)
      annotated_frame_2 = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_frame_2)
      
      annotated_frame_2 = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[team_comparison(tracks, i, 1)],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame_2)
      
      annotated_frame_2 = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[team_comparison(tracks, i, 2)],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame_2)

      annotated_frame_2 = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame_2)
      annotated_frame_2 = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_goalkeepers_xy,
        face_color=sv.Color.from_hex('000000'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame_2)
      
      annotated_frames.append(annotated_frame_2)
    return annotated_frames