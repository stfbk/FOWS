# import os
# from pathlib import Path

# folder_path = './fows_preprocessed/fows_no_occlusion'

# occ_dataset_info = {

# }

# for root, dirs, files in os.walk(folder_path):
#     if files:
#         # print(f"file path: {os.path.join(os.path.relpath(root, files[0]), files)}")
#         for file in files:
#             # print(f"file path: {os.path.join(os.path.relpath(root, file), file)}")

#             # \fows_preprocessed\fows_no_occlusion\training\user_85808\original_faces\hand_occlusion_2\frame78.jpg
#             file_path = os.path.join(os.path.relpath(root, file), file).replace('\\', '/')
#             # breakpoint()
#             img_name = file_path.split('/')[-1]
#             challenge_name = file_path.split('/')[-2]
#             algo_name = file_path.split('/')[-3]
#             user_id = file_path.split('/')[-4]
#             data_split = file_path.split('/')[-5]

#             print(f"{data_split} - {user_id} - {algo_name} - {challenge_name} - {img_name}\n")
#             breakpoint()


import json
from collections import defaultdict
import pathlib

def build_dataset_dictionary(root_folder_path):
    # Dynamic infinitely nesting dictionary setup
    auto_dict = lambda: defaultdict(auto_dict)
    dataset_dict = defaultdict(auto_dict)
    root_dir = pathlib.Path(root_folder_path)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            relative_parts = file_path.relative_to(root_dir).parts
            
            if len(relative_parts) >= 5:
                split = relative_parts[0]          
                user_id = relative_parts[1]        
                algo_name = relative_parts[2]      
                challenge_name = relative_parts[3] 
                frame_name = relative_parts[-1]    
                
                if not isinstance(dataset_dict[split][user_id][algo_name][challenge_name], list):
                    dataset_dict[split][user_id][algo_name][challenge_name] = []
                
                dataset_dict[split][user_id][algo_name][challenge_name].append(frame_name)
                
    return convert_to_dict(dataset_dict)

def convert_to_dict(d):
    """Recursively converts a nested defaultdict into standard dicts."""
    if isinstance(d, defaultdict):
        return {k: convert_to_dict(v) for k, v in d.items()}
    return d

def save_dataset_to_json(dataset_dict, output_json_path):
    """
    Saves the Python dictionary to a pretty-printed JSON file.
    """
    output_path = pathlib.Path(output_json_path)
    
    # Automatically create parent folders for the JSON file if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with UTF-8 encoding and clear indentation
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset_dict, json_file, indent=4, ensure_ascii=False)
    
    print(f"Dataset successfully exported to: {output_json_path}")

# --- Executing the Pipeline ---
if __name__ == "__main__":
    # 1. Define your path configs
    dataset_folder = './fows_preprocessed/fows_occlusion'
    output_file = "UPDATED_FOWS_OCC_frames_index.json"
    
    # 2. Run the crawler
    my_dataset_dict = build_dataset_dictionary(dataset_folder)
    
    # 3. Export to JSON file
    save_dataset_to_json(my_dataset_dict, output_file)