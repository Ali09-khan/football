from sklearn.cluster import KMeans

class TeamAssigner():
  def __init__(self):
    self.team_colors = {}
    self.player_team_dict = {}

  def get_clustering_model(self, image):
    image_2d = image.reshape(-1,3)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
    kmeans.fit(image_2d)
    return kmeans

  def get_player_color(self, frame, bbox):
    image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    top_half_image = image[0:int(image.shape[0] / 2), :]
    kmeans = self.get_clustering_model(top_half_image)
    labels = kmeans.labels_
    clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
    #getting corner clusters
    corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
    non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    player_cluster = 1 - non_player_cluster
    player_color = kmeans.cluster_centers_[player_cluster]
    return player_color

  def assign_team_color(self, frame, player_detections):
    player_colors = []
    for _, player_detection in player_detections.items():
      bbox = player_detection["bbox"]
      player_color = self.get_player_color(frame, bbox)
      player_colors.append(player_color)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
    kmeans.fit(player_colors)
    self.kmeans = kmeans
    self.team_colors[1] = kmeans.cluster_centers_[0]
    self.team_colors[2] = kmeans.cluster_centers_[1]

  def get_player_team(self, frame, player_bbox, player_id):
    if player_id in self.player_team_dict:
      return self.player_team_dict[player_id]
    player_color = self.get_player_color(frame, player_bbox)
    team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
    team_id+=1
    self.player_team_dict[player_id] = team_id
    return team_id

  def overall_color_assignment(self, video_frames, tracks):
    self.assign_team_color(video_frames[0],
                          tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
      for player_id, track in player_track.items():
        team = self.get_player_team(video_frames[frame_num],
                                    track['bbox'],
                                    player_id)

        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team]