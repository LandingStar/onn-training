
import os
import shutil

def organize_csv_files(base_dir):
    print(f"Scanning {base_dir}...")
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Iterate over experiment subdirectories
    for item in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, item)
        if os.path.isdir(exp_dir):
            csv_dir = os.path.join(exp_dir, "csv_logs")
            os.makedirs(csv_dir, exist_ok=True)
            
            moved_count = 0
            for filename in os.listdir(exp_dir):
                if filename.endswith(".csv"):
                    src = os.path.join(exp_dir, filename)
                    dst = os.path.join(csv_dir, filename)
                    shutil.move(src, dst)
                    moved_count += 1
            
            if moved_count > 0:
                print(f"Moved {moved_count} CSV files in {item} to {csv_dir}")
            else:
                # Remove empty csv_logs if no files were moved
                if not os.listdir(csv_dir):
                    os.rmdir(csv_dir)

if __name__ == "__main__":
    tasks = [
        "task1/ai_refined/results",
        "task1/ai_refined_5c/results",
        "task1/ai_refined_5_detectors/results"
    ]
    
    for task_dir in tasks:
        organize_csv_files(task_dir)
