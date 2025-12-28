import os
import subprocess

# Converts the videos to mp3
videos_dir = "videos"
audios_dir = "audios"

# Create audios directory if it doesn't exist
os.makedirs(audios_dir, exist_ok=True)

# List files in the videos directory
try:
    files = os.listdir(videos_dir)
except FileNotFoundError:
    print(f"Error: The directory '{videos_dir}' was not found.")
    files = []

for file in files:
    # Ensure we only process files (not directories) and skip the script itself
    if os.path.isfile(os.path.join(videos_dir, file)):
        # Assumes filenames are simple like 'Lecture 2.mkv' and you want 'Lecture 2' as the name
        # The original code's complex splitting logic caused the IndexError.
        
        file_name_without_ext, ext = os.path.splitext(file)
        
        # If the file names are 'Lecture X.mkv', a simple approach to get the number 'X'
        try:
            # Example: from 'Lecture 2', get '2'
            tutorial_number = file_name_without_ext.split(' ')[1] 
        except IndexError:
            # Fallback if the name doesn't contain a space
            tutorial_number = file_name_without_ext
            
        file_name = file_name_without_ext

        print(f"Processing: {file}")
        print(f"  Tutorial Number: {tutorial_number}, File Name: {file_name}")

        # The f-string must be properly quoted for shell execution
        # Use a list of arguments for better security and handling of spaces in names
        input_path = os.path.join(videos_dir, file)
        output_path = os.path.join(audios_dir, f"{tutorial_number}_{file_name}.mp3")

        # The command uses ffmpeg to convert video to mp3 (audio only)
        command = [
            "ffmpeg",
            "-i", input_path,
            "-vn", # Disable video recording
            "-acodec", "libmp3lame", # Use libmp3lame for MP3 encoding
            "-q:a", "2", # Variable bitrate quality (2 is high quality)
            output_path
        ]
        
        try:
            # Execute the ffmpeg command
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("  Conversion successful.")
        except subprocess.CalledProcessError as e:
            print(f"  Conversion failed for {file}. Error:")
            print(e.stderr)
        except FileNotFoundError:
            print("  Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            break # Stop processing if ffmpeg isn't found