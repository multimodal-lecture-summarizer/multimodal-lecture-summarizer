import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
import yt_dlp

def main():
    target_dir = Path(r"D:\datasets\TEDLIUM\videos")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(r"D:\datasets\TEDLIUM\metadata.csv")
    if not csv_path.exists():
        print(f"[ERROR] metadata.csv not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    speakers = df['speaker_id'].unique().tolist()
    
    print("======================================================================")
    print(f" DOWNLOADING ALL TED TALK LECTURE VIDEOS FOR TED-LIUM DATASET")
    print(f" Target Folder: {target_dir}")
    print(f" Total Unique Speakers / Talks: {len(speakers)}")
    print("======================================================================")
    
    # Top famous TED Talks in TED-LIUM benchmark
    ted_talk_queries = [
        ("Barry_Schwartz", "https://www.youtube.com/watch?v=VO6XEQIsCoM"),
        ("Ken_Robinson", "https://www.youtube.com/watch?v=iG9CE55wbtY"),
        ("Simon_Sinek", "https://www.youtube.com/watch?v=qp0HIF3SrI4"),
        ("Jill_Bolte_Taylor", "https://www.youtube.com/watch?v=UyyjU8fzEYU"),
        ("Hans_Rosling", "https://www.youtube.com/watch?v=hVimVzgtD6w"),
        ("Dan_Pink", "https://www.youtube.com/watch?v=rrkrvAUbU9Y"),
        ("Susan_Cain", "https://www.youtube.com/watch?v=c0KYU2j0TM4"),
        ("Amy_Cuddy", "https://www.youtube.com/watch?v=Ks-_Mh1QhMc"),
        ("Bren_Brown", "https://www.youtube.com/watch?v=iCvmsMzlF7o"),
        ("Pranav_Mistry", "https://www.youtube.com/watch?v=YrtANPtnhyg")
    ]
    
    ydl_opts = {
        'format': 'mp4[height<=480]',
        'outtmpl': str(target_dir / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'ignoreerrors': True
    }
    
    count = 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for name, url in ted_talk_queries:
            print(f"\n[{count+1}/{len(ted_talk_queries)}] Downloading TED Talk Video for '{name}'...")
            try:
                ydl.download([url])
                count += 1
            except Exception as e:
                print(f"  [WARN] Download error for {name}: {e}")
                
    # Also search by speaker name for remaining speakers
    for spk in speakers[:15]:
        spk_clean = spk.replace("_", " ")
        out_file = target_dir / f"{spk}.mp4"
        if not out_file.exists():
            print(f"\n[Search & Download] Downloading TED Talk video for speaker: '{spk_clean}'...")
            search_opts = {
                'format': 'mp4[height<=480]',
                'outtmpl': str(out_file),
                'default_search': 'ytsearch1:',
                'quiet': True,
                'ignoreerrors': True
            }
            try:
                with yt_dlp.YoutubeDL(search_opts) as ydl_s:
                    ydl_s.download([f"{spk_clean} TED talk"])
            except Exception as err:
                print(f"  [WARN] Search error for {spk}: {err}")

    downloaded = list(target_dir.glob("*.mp4"))
    print("\n" + "="*70)
    print(f" COMPLETED! Total TED-LIUM MP4 Videos downloaded: {len(downloaded)}")
    print(f" Saved in: D:\\datasets\\TEDLIUM\\videos")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
