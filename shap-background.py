import torch
from tqdm import tqdm
from torch.autograd import Variable
import random
import json
import pandas as pd
from os import listdir
from os.path import join, dirname, basename, exists
from tqdm import tqdm
import yaml

from utils.predict_helpers import coords_raw_to_norm, get_frame_rate, get_video_metadata, get_data, load_data_all_videos, median_filter, read_csv_to_array


def setup(tracking_coords=0, frame_rate=0, folder_path="", video="", window_stride: int=2):

    if video:
        video_path = join(folder_path, video)

        # Get video metadata
        _, _, total_frames = get_video_metadata(video_path)
        frame_rate = get_frame_rate(video_path) # Get float value of frame_rate
        
        # Get tracker data
        tracker_data_file_path = join(dirname(video_path), 'outputs', basename(video_path).split('.')[0] + "_coordinates.csv")
        if exists(tracker_data_file_path):
            tracking_coords = read_csv_to_array(tracker_data_file_path)
            tracking_coords = np.array(tracking_coords)

    # Resample skeleton sequence
    pred_frame_rate = 30.0
    
    # Seconds between each prediction window
    pred_interval_seconds = 2.5

    # Create header for csv
    body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow',
                'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip',
                'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
    
    num_joints = len(body_parts)
    num_channels = 2
    tracking_coords = tracking_coords.astype(float)
    num_resampled_frames = np.interp(np.arange(0, len(tracking_coords[:,0]), (frame_rate / pred_frame_rate)), np.arange(0, len(tracking_coords[:,0])), tracking_coords[:,0]).shape[0] #int((total_frames/frame_rate)*pred_frame_rate)
    resampled_tracking_coords = np.ndarray(shape=(num_resampled_frames, num_joints, num_channels))
    for j in range (0, num_joints*2, num_channels):
        resampled_tracking_coords[:,j//num_channels,0] = np.interp(np.arange(0, len(tracking_coords[:,j]), (frame_rate / pred_frame_rate)), np.arange(0, len(tracking_coords[:,j])), tracking_coords[:,j])
        resampled_tracking_coords[:,j//num_channels,1] = np.interp(np.arange(0, len(tracking_coords[:,j+1]), (frame_rate / pred_frame_rate)), np.arange(0, len(tracking_coords[:,j+1])), tracking_coords[:,j+1]) 
        
    # Filter, centralize and normalize coordinates
    pelvis_xs = resampled_tracking_coords[:,body_parts.index('pelvis'),0]
    pelvis_ys = resampled_tracking_coords[:,body_parts.index('pelvis'),1]
    thorax_xs = resampled_tracking_coords[:,body_parts.index('thorax'),0]
    thorax_ys = resampled_tracking_coords[:,body_parts.index('thorax'),1]
    median_pelvis_x = np.median(pelvis_xs)
    median_pelvis_y = np.median(pelvis_ys)
    trunk_lengths = np.sqrt((thorax_xs - pelvis_xs)**2 + (thorax_ys - pelvis_ys)**2)
    median_trunk_length = np.median(trunk_lengths)

    # Apply median filter
    filter_coords = median_filter(resampled_tracking_coords, window_stride)
    
    # Centralize and normalize
    norm_coords = coords_raw_to_norm(filter_coords, median_pelvis_x, median_pelvis_y, median_trunk_length)
    data = np.expand_dims(norm_coords, 0)

    return data

def get_model_and_data(data, num_models, num_portions):
    
    # Perform inference per model
    models = [i for i in range(1, num_models+1)]
    portions = ['' if i==1 else i for i in range(1, num_portions+1)]

    ensemble_models = []
    ensemble_data = []
    for model in models:
        for portion in portions:
            config_file = join('models', 'configs', 'gcn_5Best{0}_randinit_train{1}.yaml'.format(model, portion))
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            args = {
                **config['directories'], 
                **config['train_feeder'], 
                **config['val_feeder'], 
                **config['model'], 
                **config['graph'],
                **config['features'], 
                **config['optimization'], 
                **config['training'], 
                **config['validation']
            }
            args['val_parts_distance'] = 75
            if torch.cuda.is_available(): #GPU
                args['val_batch_size'] = 32
            else: #CPU
                args['val_batch_size'] = 4
            args['num_workers'] = 0#4
            weights_path = join('models', args['weights'])

            eval_data, model2, target_layer, output_device = get_data(data, weights_path, args, 'shap')
            
            ensemble_data = eval_data
            ensemble_models.append(model2)

    return ensemble_models, ensemble_data, output_device

def shap_background_data(folder_path):
    """ Create background data for SHAP explainer. """

    df = pd.read_excel(folder_path + "/jama_coordinates.xlsx")

    num_videos = 50
    num_epochs = num_videos * 12
    num_CP = round(num_epochs * 0.15)
    num_non_CP = num_epochs - num_CP

    folder_path = folder_path + "/jama_trainval"

    file_names = listdir(folder_path)
    indexes = list(range(len(file_names)))
    selected_indexes = random.sample(indexes, num_videos)

    selected_videos = []
    graph_data = []
    cp_overview = []

    for i in selected_indexes:
        selected_videos.append(file_names[i].replace("tracked_", "").replace(".csv", ""))
        video_info = df[df["Video ID"] == file_names[i].replace("tracked_", "").replace(".csv", "")].squeeze()
        if video_info.empty:
            print(file_names[i])
        frame_rate = video_info["FPS"]
        CP_inf = video_info["CP"] # Yes or No
        tracking_coords = read_csv_to_array(folder_path + "/" + file_names[i])
        tracking_coords = np.array(tracking_coords)
        data = setup(tracking_coords=tracking_coords, frame_rate=frame_rate)
        ensemble_models, ensemble_data, output_device = get_model_and_data(data, 1, 1)
        graph_data.append(ensemble_data)
        cp_overview.append(True if CP_inf == "Yes" else False)

    print("Number of videos in background data:", len(graph_data))
    num_CP_videos = cp_overview.count(True)
    num_non_CP_videos = cp_overview.count(False)

    epochs_CP = round(num_CP/num_CP_videos)
    epochs_non_CP = round(num_non_CP/num_non_CP_videos)

    concat_data = load_data_all_videos(graph_data, cp_overview, epochs_CP, epochs_non_CP)

    video_info_df = df[df["Video ID"].isin(selected_videos)]
    video_info_df.to_excel("background_data_info.xlsx", index=False)
    
    process = tqdm(concat_data)

    for batch_idx, (data_y, index) in enumerate(process):
        with torch.enable_grad():
            # Fetch batch
            if torch.cuda.is_available(): #GPU
                data_y = Variable(
                    data_y.float().cuda(output_device),
                    requires_grad=True)
            else: #CPU
                data_y = Variable(
                    data_y.float(),
                    requires_grad=True)
        
        data_list = data_y.tolist()
        json_data = json.dumps(data_list, indent=4)
        with open("background_data.json", "w") as file_data:
            file_data.write(json_data)

