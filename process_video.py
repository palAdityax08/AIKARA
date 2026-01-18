import os
import subprocess

# Convert the videos to mp3
videos_dir = "videos"
audios_dir = "audios"

os.makedirs(audios_dir, exist_ok=True)

try:
    files = os.listdir(videos_dir)
except FileNotFoundError:
    print(f"Error: The directory '{videos_dir}' was not found.")
    files = []

for file in files:
    if os.path.isfile(os.path.join(videos_dir, file)):    
        file_name_without_ext, ext = os.path.splitext(file)
        try:
            tutorial_number = file_name_without_ext.split(' ')[1] 
        except IndexError:
            tutorial_number = file_name_without_ext
            
        file_name = file_name_without_ext

        print(f"Processing: {file}")
        print(f"  Tutorial Number: {tutorial_number}, File Name: {file_name}")
        input_path = os.path.join(videos_dir, file)
        output_path = os.path.join(audios_dir, f"{tutorial_number}_{file_name}.mp3")

        command = [
            "ffmpeg",
            "-i", input_path,
            "-vn", 
            "-acodec", "libmp3lame", 
            "-q:a", "2", 
            output_path
        ]
        
        try:
    
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("  Conversion successful.")
        except subprocess.CalledProcessError as e:
            print(f"  Conversion failed for {file}. Error:")
            print(e.stderr)
        except FileNotFoundError:
            print("  Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            break 
