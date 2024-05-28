import math
import os
from os.path import join, dirname, basename, exists
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils.predict_helpers import get_shap_color

class SHAP_visualization():

    def __init__(self, shap_values):
        self.shap_values = [np.array(s) for s in shap_values]
        self.shaps_one_class = self.shap_values[1] - self.shap_values[0] # CP - no CP
        self.shaps_mean = np.mean(self.shaps_one_class, axis=2)  # mean over the 150 frames
    
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

    def make_video_visualization(self, video_path, tracking_coords, window_preds, groups, split_point):
        pred_interval_seconds = 2.5 # Seconds between each prediction window
        shap_interval_seconds = 5.0 # Seconds between each SHAP update
        preds_per_shap_interval = int(shap_interval_seconds / pred_interval_seconds)

        shaps_dimensions_average, shaps_max_avg = self.calculate_shaps_average_x_y()

        # Load raw video
        from skvideo.io import vreader, ffprobe, FFmpegWriter
        videogen = vreader(video_path)
        video_metadata = ffprobe(video_path)['video']
        fps = video_metadata['@r_frame_rate']
        fps_num, fps_den = fps.split('/')
        fps_float = float(fps_num) / float(fps_den)
        frame_height, frame_width = next(vreader(video_path)).shape[:2]
        frame_side = frame_width if frame_width >= frame_height else frame_height

        # Initialize annotated video
        vcodec = 'libvpx-vp9'  # 'libx264'
        writer = FFmpegWriter(
            join(dirname(video_path), 'outputs', basename(video_path).split('.')[0] + '_shap.mp4'),
            inputdict={'-r': fps},
            outputdict={'-r': fps, '-bitrate': '-1', '-vcodec': vcodec, '-pix_fmt': 'yuv420p', '-lossless': '1'})

        # Annotate video
        from PIL import Image, ImageDraw
        i = 0
        while True:

            try:
                frame = next(videogen)
                image = Image.fromarray(frame)
                image_draw = ImageDraw.Draw(image)
                image_coordinates = tracking_coords[i, ...].reshape((int(tracking_coords.shape[1] / 2), 2)).astype(
                    'float32')
                window_index = math.floor((i / fps_float) / shap_interval_seconds) * preds_per_shap_interval
                window_indices = [window_index if window_index < window_preds.shape[0] else window_preds.shape[0] - 1]
                for j in range(1, preds_per_shap_interval):
                    next_window_index = window_indices[0] + j
                    if next_window_index < window_preds.shape[0]:
                        window_indices.append(next_window_index)

                # Video with SHAP values
                image_shaps = np.median(shaps_dimensions_average[window_indices], axis=0)
                colors = ["green", "white", "red"]
                cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                image = self.display_segments_shap(image, image_draw, image_coordinates, image_shaps, shaps_max_avg, cmap, image_height=frame_height, image_width=frame_width, segment_width=int(frame_side/200))
                image = self.display_body_parts_shap(image, image_draw, image_coordinates, image_shaps, shaps_max_avg, cmap, groups, image_height=frame_height, image_width=frame_width, marker_radius=int(frame_side/150), split_point=split_point)
                writer.writeFrame(np.array(image))
                i += 1
            
            except:
                break

        writer.close()
    
    def display_body_parts_shap(image, image_draw, coordinates, shaps, shaps_max_avg, cmap, groups, image_height=1024,
                            image_width=1024, marker_radius=5, split_point=False):   
        """
        Draw markers on predicted body part locations.
        
        Args:
            image: PIL Image
                The loaded image the coordinate predictions are inferred for
            image_draw: PIL ImageDraw module
                Module for performing drawing operations
            coordinates: Numpy array
                Predicted body part coordinates in image
            cams: Numpy array
                Predicted body part contribution (CAM) in image
            groups: Array
                Body keypoint indices associated with groups of body keypoints
            image_height: int
                Height of image
            image_width: int
                Width of image
            marker_radius: int
                Radius of marker
            cam_threshold: float
                Threshold value of CAM for body part to contribute towards prediction of CP
            binary: float
                Flag for two-color visualization
            color_scheme: string
                Combination of colors to use for CAM visualization (e.g., 'GYOR' for 'G' = green, 'Y' = yellow, 'O' = orange and 'R' = red)
        Returns:
            Instance of PIL image with annotated body part predictions.
        """

        # Draw markers
        shaps_pos = shaps[0]
        shaps_vel = shaps[1]

        for i, (body_part_x, body_part_y) in enumerate(coordinates):
            for group in groups:
                if i in group:
                    group_indices = np.array(group)
                    break
            
            shap_value_pos = np.mean(shaps_pos[group_indices])
            color_pos = get_shap_color(shap_value_pos, shaps_max_avg, cmap)

            shap_value_vel = np.mean(shaps_vel[group_indices])
            color_vel = get_shap_color(shap_value_vel, shaps_max_avg, cmap)
            
            body_part_x *= image_width
            body_part_y *= image_height
            
            if split_point:
                marker_radius_inner = marker_radius * 0.8
                marker_radius_outer = marker_radius * 1.5
                image_draw.ellipse([(body_part_x - marker_radius_outer, body_part_y - marker_radius_outer),
                                    (body_part_x + marker_radius_outer, body_part_y + marker_radius_outer)], fill=color_vel)
                image_draw.ellipse([(body_part_x - marker_radius_inner, body_part_y - marker_radius_inner),
                                    (body_part_x + marker_radius_inner, body_part_y + marker_radius_inner)], fill=color_pos)
            else:
                image_draw.ellipse([(body_part_x - marker_radius, body_part_y - marker_radius), (body_part_x + marker_radius, body_part_y + marker_radius)], fill=color_vel)

        return image


    def display_segments_cam(self, image, image_draw, coordinates, image_height=1024, image_width=1024, segment_width=3):
        """
        Draw segments between body parts according to predicted body part locations.
        
        Args:
            image: PIL Image
                The loaded image the coordinate predictions are inferred for
            image_draw: PIL ImageDraw module
                Module for performing drawing operations
            coordinates: Numpy array
                Predicted body part coordinates in image
            image_height: int
                Height of image
            image_width: int
                Width of image
            segment_width: int
                Width of association line between markers
            
        Returns:
            Instance of PIL image with annotated body part segments.
        """

        # Define segments and colors
        segments = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 8), (8, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12),
                    (12, 13), (13, 14), (14, 15), (12, 16), (16, 17), (17, 18)]
        segment_color = '#5c5a5a'

        # Draw segments
        for (body_part_a_index, body_part_b_index) in segments:
            body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
            body_part_a_x *= image_width
            body_part_a_y *= image_height
            body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
            body_part_b_x *= image_width
            body_part_b_y *= image_height
            image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_color,
                            width=segment_width)

        return image

    
    def display_segments_shap(self, image, image_draw, coordinates, shaps, shaps_max_avg, cmap, image_height=1024,
                          image_width=1024, segment_width=3):
        """
        Draw segments between body parts according to predicted body part locations with SHAP color.
        
        Args:
            image: PIL Image
                The loaded image the coordinate predictions are inferred for
            image_draw: PIL ImageDraw module
                Module for performing drawing operations
            coordinates: Numpy array
                Predicted body part coordinates in image
            image_height: int
                Height of image
            image_width: int
                Width of image
            segment_width: int
                Width of association line between markers
            
        Returns:
            Instance of PIL image with annotated body part segments.
        """

        # Define segments and colors
        neighbor_link = [(0, 1), (2, 1), (3, 1), (1, 4), (9, 8), (10, 9), (11, 10), (5, 8), (6, 5), (7, 6), (4, 8), (12, 8),
                        (16, 12), (17, 16), (18, 17), (13, 12), (14, 13), (15, 14)]
        segment_color = '#5c5a5a'

        shaps_bones = shaps[2]

        # Draw segments
        for (body_part_a_index, body_part_b_index) in neighbor_link:
            body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
            body_part_a_x *= image_width
            body_part_a_y *= image_height
            body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
            body_part_b_x *= image_width
            body_part_b_y *= image_height

            shap_value = shaps_bones[body_part_a_index]
            segment_color = get_shap_color(shap_value, shaps_max_avg, cmap)

            image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_color,
                            width=segment_width)

        return image


    def get_shap_color(self, shap_value, shaps_max_avg, cmap):
        norm = Normalize(vmin=-shaps_max_avg, vmax=shaps_max_avg)
        normalized_shap = norm(shap_value)
        color = cmap(normalized_shap)

        # Format the integers into a hexadecimal string
        rgb_integers = [int(x * 255) for x in color[:3]]
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_integers)

        return hex_color
