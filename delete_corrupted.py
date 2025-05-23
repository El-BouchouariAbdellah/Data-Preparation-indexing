import os
import shutil

def delete_corrupted_subfolders(parent_folder):
    if not os.path.isdir(parent_folder):
        print(f"Error: '{parent_folder}' is not a valid directory.")
        return

    deleted_any = False
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name.endswith('_corrupted'):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
                deleted_any = True
            except Exception as e:
                print(f"Failed to delete {folder_path}: {e}")
    
    if not deleted_any:
        print("No '_corrupted' folders found to delete.")

if __name__ == "__main__":
    user_input = input("Enter the path to the parent folder: ").strip()
    delete_corrupted_subfolders(user_input)
