import subprocess
import os

def extract_audio(video_path, output_path=None):
    try:
        # If output path is not specified, create one based on input file
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + '.mp3'
        
        # Construct FFmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-q:a', '2',  # High quality audio
            output_path
        ]
        
        # Execute FFmpeg command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Audio successfully extracted to: {output_path}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

if __name__ == "__main__":
    video_path = "/home/dukhanin/xmas_hack/data/Презентация_Кейс1.mp4"
    extract_audio(video_path) 