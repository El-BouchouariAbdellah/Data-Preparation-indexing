import os
import shutil
from pathlib import Path

def move_all_files(source_folder, destination_folder):
    """
    Move all files from source folder and all its subfolders to destination folder.
    Creates destination folder if it doesn't exist.
    """
    # Convert to Path objects for easier handling
    source = Path(source_folder)
    destination = Path(destination_folder)
    
    # Check if source folder exists
    if not source.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return False
    
    if not source.is_dir():
        print(f"Error: '{source_folder}' is not a directory.")
        return False
    
    # Create destination folder if it doesn't exist
    destination.mkdir(parents=True, exist_ok=True)
    
    # Get all files recursively from source folder and all subfolders
    files = []
    for root, dirs, filenames in os.walk(source):
        for filename in filenames:
            files.append(Path(root) / filename)
    
    if not files:
        print(f"No files found in '{source_folder}' or its subfolders.")
        return True
    
    moved_count = 0
    failed_moves = []
    
    for file_path in files:
        try:
            # Destination file path
            dest_file = destination / file_path.name
            
            # Move the file directly (no conflict handling)
            shutil.move(str(file_path), str(dest_file))
            print(f"Moved: {file_path.relative_to(source)} -> {dest_file.name}")
            moved_count += 1
            
        except Exception as e:
            failed_moves.append(f"{file_path.name}: {str(e)}")
            print(f"Failed to move {file_path.name}: {e}")
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"Successfully moved: {moved_count} files")
    if failed_moves:
        print(f"Failed moves: {len(failed_moves)}")
        for failure in failed_moves:
            print(f"  - {failure}")
    
    return len(failed_moves) == 0

# Example usage
if __name__ == "__main__":
    # Your specific paths
    source_folder = r"C:\Users\abdel\Desktop\السادس ابتدائي\النشاط العلمي\مصادر الطاقة المستوى السادس ابتدائي"
    destination_folder = r"C:\Users\abdel\Desktop\السادس ابتدائي\النشاط العلمي"
    
    print(f"Moving all files from: {source_folder}")
    print(f"To: {destination_folder}")
    print("This will move files from ALL subfolders within Grammaire folder.\n")
    
    confirm = input("Do you want to proceed? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        success = move_all_files(source_folder, destination_folder)
        
        if success:
            print("\nAll files moved successfully!")
        else:
            print("\nSome files could not be moved. Check the errors above.")
    else:
        print("Operation cancelled.")