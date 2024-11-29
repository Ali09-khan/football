import numpy as np

def return_pitch_xy(tracks, obj, frame_num):
    frame_data = tracks[obj][frame_num]
    positions = []

    for key, value in frame_data.items():
        if "position" in value:
            if value["position"] is not None:
                positions.append(value["position"])
            else:
                positions.append(np.array([]))

    return np.array(positions)

def team_comparison(tracks, frame_num, team_id):
    frame_data = tracks["players"][frame_num]
    print(frame_data)
    return np.array([value["team"] == team_id for key, value in frame_data.items()])