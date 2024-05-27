import json
import os
from os import listdir
from os.path import join, dirname, basename, exists
import random
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import yaml
import torch
from zipfile import ZipFile 
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils.predict_helpers import coords_raw_to_norm, get_frame_rate, get_shap_color, get_video_metadata, get_data, load_data_all_videos, median_filter, read_csv_to_array

class SHAP():

    def __init__(self, shap_values):
        self.shap_values = [np.array(s) for s in shap_values]
        self.shaps_one_class = self.shap_values[1] - self.shap_values[0] # CP - no CP
        self.shaps_mean = np.mean(self.shaps_one_class, axis=2)  # mean over the 150 frames

    def get_shaps_mean(self):
        return self.shaps_mean
    
    def calculate_shaps_average_x_y(self, shap_list):
        """ Calculate SHAP values for x and y coordinates per feature type """

        shaps_pos = np.mean(self.shaps_mean[:, 0:2, :], axis=1)
        shaps_vel = np.mean(self.shaps_mean[:, 2:4, :], axis=1)
        shaps_bones = np.mean(self.shaps_mean[:, 4:6, :], axis=1)
        shaps_dimensions_average = np.stack((shaps_pos, shaps_vel, shaps_bones), axis=1)
        shaps_max_avg = np.max(np.abs(shaps_dimensions_average))

        return shaps_dimensions_average, shaps_max_avg

    def visualize_shap_values(self, windows, heatmap, summary_plot, mean_plots, body_parts, video_path, data_video):
        """ Create heatmap and summary plots of shap values """

        dimensions = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'bones_x', 'bones_y']
        max_shap_value = self.shaps_mean.max()
        min_shap_value = self.shaps_mean.min()
        color_range = max(abs(max_shap_value), abs(min_shap_value))

        for window in range(windows):  # for each window
            shap_per_window = self.shaps_mean[window, :, :]
            colors = ["green", "white", "red"]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

            if heatmap:
                self.make_heatmap(shap_per_window, window, body_parts, cmap, dimensions, color_range, video_path)
                
            if summary_plot:
                self.make_summary_plot(shap_per_window, data_video, window, body_parts, video_path)

        if mean_plots:
            self.make_mean_plots(self.shaps_mean, data_video, body_parts, dimensions, color_range, cmap, video_path)



    def make_heatmap(self, shap_per_window, time, body_parts, cmap, dimensions, color_range, video_path):
        """
        Make heatmap of SHAP values for a given window.
        
        Args:
            shap_per_window: ndarray
                SHAP values for the window
            time: int
                Window index
            body_parts: list
                List of body parts
            cmap: LinearSegmentedColormap
                Color map for the heatmap
            dimensions: list
                List of dimensions
            color_range: float
                Maximum range of SHAP values
            video_path: string
                Path to video
        """

        # Sort the shap values
        row_sums = np.abs(shap_per_window).sum(axis=0)
        sorted_rows_asc = np.argsort(row_sums)
        sorted_rows = sorted_rows_asc[::-1]  # Sort the rows in descending order
        sorted_shap_values = shap_per_window[:, sorted_rows]  # Sort the shap_values
        sorted_body_parts = [body_parts[i] for i in sorted_rows]  # Sort the body parts

        sorted_shap_values_transpose = sorted_shap_values.T
        # Transpose the array to get body points on the y-axis and dimensions on the x-axis

        plt.figure(figsize=(10, 10))
        heatmap = sns.heatmap(sorted_shap_values_transpose, cmap=cmap, xticklabels=dimensions,
                                yticklabels=sorted_body_parts, vmin=-color_range, vmax=color_range,
                                linecolor="black", annot=True, fmt=".4f", linewidths=0.008, cbar_kws={
                'label': 'SHAP value (impact on model output)'})
        # Use annot=True, fmt=".4f" to show shap_values in plot
        heatmap.xaxis.tick_top()  # x axis on top
        plt.title("Video " + basename(video_path).split('.')[0] + " window " + str(time), fontsize=20)

        # Save plots
        folder = join(dirname(video_path), 'outputs', 'shap', 'image_plot', basename(video_path).split('.')[0])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(join(folder, basename(video_path).split('.')[0] + '_' + str(time) + '_shap.png'), format='png')
        plt.close()

    def make_summary_plot(self, shap_per_window, data_video, time, body_parts, video_path):
        """
        Make summary_plot of SHAP values for a given window.
        
        Args:
            shap_per_window: ndarray
                SHAP values for the window
            data_video: ndarray
                Video data
            time: int
                Window index
            body_parts: list
                List of body parts
            video_path: string
                Path to video
        """
        data_per_timeframe = data_video[time, :, :, :]

        data_per_timeframe = data_per_timeframe.cpu().detach().numpy()
        data_per_window = np.mean(data_per_timeframe, axis=1)  # mean over the 150 frames

        shap.summary_plot(shap_per_window, data_per_window, show=False, feature_names=body_parts)
        plt.title("Video " + basename(video_path).split('.')[0] + " window " + str(time), y=0.99)

        folder = join(dirname(video_path), 'outputs', 'shap', 'summary_plot',
                    basename(video_path).split('.')[0])

        os.makedirs(folder, exist_ok=True)
        plt.savefig(join(folder, basename(video_path).split('.')[0] + '_' + str(time) + '_shap.png'), format='png')
        plt.close()

    def make_mean_plots(self, shaps_mean, data_video, body_parts, dimensions, color_range, cmap, video_path):
        """
        Make summary plot and heatmap of mean SHAP values.
        
        Args:
            shaps_mean: ndarray
                Mean SHAP values for all windows
            data_video: ndarray
                Video data
            body_parts: list
                List of body parts
            dimensions: list
                List of dimensions
            color_range: float
                Maximum range of SHAP values
            cmap: LinearSegmentedColormap
                Color map for the heatmap
            video_path: string
                Path to video
        """
        shaps_overall_mean = np.mean(shaps_mean, axis=0)  # median over all windows

        # Sort the shap values
        row_sums = np.abs(shaps_overall_mean).sum(axis=0)
        sorted_rows_asc = np.argsort(row_sums)
        sorted_rows = sorted_rows_asc[::-1]  # Sort the rows in descending order
        sorted_shap_values = shaps_overall_mean[:, sorted_rows]  # Sort the shap_values
        sorted_body_parts = [body_parts[i] for i in sorted_rows]  # Sort the body parts

        sorted_shap_values_transpose = sorted_shap_values.T
        # Transpose the array to get body points on the y-axis and dimensions on the x-axis

        plt.figure(figsize=(10, 10))
        heatmap = sns.heatmap(sorted_shap_values_transpose, cmap=cmap, annot=True, fmt=".4f", xticklabels=dimensions,
                                yticklabels=sorted_body_parts, vmin=-color_range, vmax=color_range, linecolor="black",
                                linewidths=0.008, cbar_kws={
                'label': 'SHAP value (impact on model output)'})
        # Use annot=True, fmt=".4f" to show shap_values in plot
        heatmap.xaxis.tick_top()  # x axis on top
        plt.title("Video " + basename(video_path).split('.')[0] + " mean of all windows", fontsize=20)
        folder = join(dirname(video_path), 'outputs', 'shap', 'image_plot', basename(video_path).split('.')[0])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(join(folder, basename(video_path).split('.')[0] + '_mean_shap.png'), format='png')
        plt.close()

        data_video = data_video.cpu().detach().numpy()
        data_video_mean = np.mean(data_video, axis=2)  # mean over all timeframes
        data_video_overall_mean = np.mean(data_video_mean, axis=0)  # median over all windows

        shap.summary_plot(shaps_overall_mean, data_video_overall_mean, show=False, feature_names=body_parts)
        plt.title("Video " + basename(video_path).split('.')[0] + " mean of all windows", y=0.99)
        folder = join(dirname(video_path), 'outputs', 'shap', 'summary_plot', basename(video_path).split('.')[0])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(join(folder, basename(video_path).split('.')[0] + '_mean_shap.png'), format='png')
        plt.close()

    def make_skeleton_image(self, sample_coords, sample_conns, num_body_parts, groups, shaps_dimensions_average, shaps_max_avg, video_path, split_point):
        """ Create skeleton image visualiztion using SHAP values."""

        # Initialize visualization
        location = np.expand_dims(np.expand_dims(np.swapaxes(np.asarray(sample_coords), 0, 1), 1), -1)
        location[1, ...] = 1.0 - location[1,...]
        location[0, ...] *= 1080
        location[1, ...] *= 1920
        
        # Store overall SHAP visualization
        plt.figure()
        plt.ion()
        plt.cla()
        plt.xlim(-100, 1180)
        plt.ylim(220, 1500)
        plt.axis('off')
        x = location[0, 0, :, 0]
        y = location[1, 0, :, 0]
        connections = np.asarray(sample_conns) + 1
        c_pos = []
        c_vel = []
        for v in range(num_body_parts):
            for group in groups:
                if v in group:
                    group_indices = np.array(group)
                    break
            
            shap_value = np.mean(shaps_dimensions_average[:, :, group_indices], axis=0)
            shap_value_pos = np.mean(shap_value[0])
            shap_value_vel = np.mean(shap_value[1])
            shap_value_bones = np.mean(shap_value[2])
            colors = ["green", "white", "red"]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
            
            shap_vel_color = get_shap_color(shap_value_vel, shaps_max_avg, cmap)
            shap_pos_color = get_shap_color(shap_value_pos, shaps_max_avg, cmap)

            k = connections[v] - 1
            shap_bones_color = get_shap_color(shap_value_bones, shaps_max_avg, cmap)

            plt.plot([x[v], x[k]], [y[v], y[k]], '-o', c=shap_bones_color, linewidth=3.0, markersize=0, zorder=-1)
            c_pos.append(shap_pos_color)
            c_vel.append(shap_vel_color)
        if split_point:
            plt.scatter(x, y, marker='o', c=c_vel, s=140, edgecolors='black', zorder=0)
            plt.scatter(x, y, marker='o', c=c_pos, s=40, zorder=1)
        else:
            plt.scatter(x, y, marker='o', c=c_vel, s=100, edgecolors='black', zorder=1)
        plt.ioff()
        plt.savefig(
        join(dirname(video_path), 'outputs', basename(video_path).split('.')[0] + '_shap.png'),
        format='png')





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

def shap_background_data_without_random(folder_path, video_names, num_models: int=10, num_portions: int=7):
    graph_data = []
    for video in video_names:

        data = setup(folder_path=folder_path, video=video)
        ensemble_models, video_data, output_device = get_model_and_data(data, num_models, num_portions)
        graph_data.append(video_data)

    print("Number of videos in background data:", len(graph_data))

    # Slå sammen flere videoer
    concat_data = load_data_all_videos(graph_data)
    
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
        
        extracted_file = "background_data.json"
        with open(extracted_file, "w") as file_data:
            file_data.write(json_data)
        
        with ZipFile("background_data.zip", 'w') as zip_file:
            zip_file.write(extracted_file)

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

