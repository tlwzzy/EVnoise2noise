import os
import cv2
import glob
import pyrootutils
import rootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
path = rootutils.find_root(search_from=__file__, indicator=".project-root")

train_path = path / "data" / "train"
val_path = path / "data" / "valid"
test_path = path / "data" / "test"

def preprocess_images(directory: str) -> None:
    # Get all PNG files in the directory
    image_paths = glob.glob(os.path.join(directory, '*.png'))

    for image_path in image_paths:
        # Load the image
        img = cv2.imread(image_path)

        # Replace white (also shades of whites) pixels with black pixels
        # img[(img == [255, 255, 255]).all(axis=2)] = [0, 0, 0]
        img[(img > [128, 128, 128]).all(axis=2)] = [0, 0, 0]

        # Create a new path for the preprocessed image
        base_dir, filename = os.path.split(image_path)
        new_filename = os.path.join(base_dir, 'preprocessed', filename)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)

        # Save the preprocessed image
        cv2.imwrite(new_filename, img)

    print("Preprocessing completed.")
preprocess_images(test_path)
# preprocess_images(val_path)
# preprocess_images(train_path)

