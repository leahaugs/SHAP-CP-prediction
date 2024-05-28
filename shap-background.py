import torch
from tqdm import tqdm
from torch.autograd import Variable
import random
import json
import pandas as pd
from os import listdir
from tqdm import tqdm

from utils.predict_helpers import load_data_all_videos, read_csv_to_array, setup, get_model_and_data

def shap_background_data(folder_path):
    """ 
    Create background data for SHAP explainer. 
    
    Args:
        folder_path: string
            Path to folder with tracker data
    """

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
        _, ensemble_data, output_device = get_model_and_data(data, 1, 1)
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

    for _, (data_y, _) in enumerate(process):
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

