# 12-Lead ECG Dataset utils for Real-Time Lead Detection using YOLO 

This repository provides resources and scripts to prepare and augment 12-lead ECG datasets for use with YOLOv7 models. The scripts include utilities for crawling background images, augmenting ECG data with various effects, extracting Regions of Interest (ROIs), and managing dataset organization.

---

### Examples (YOLO Dataset)
![nn](https://ifh.cc/g/vFKtT8.png)
![nn](https://ifh.cc/g/zwsYWa.png)
![nn](https://ifh.cc/g/3doOZ0.png)
![nn](https://ifh.cc/g/4B1jcW.png)

---

## Requirements
To run the scripts in this repository, install the following dependencies:

### Install Requirements
To install the dependencies on Ubuntu, use the following command:
```bash
pip install -r requirements.txt
```

Ensure you have Python and `pip` installed on your system. You can verify this with:
```bash
python --version
pip --version
```

---

## Contents

### `crawling_bg.py`
This script is used to download background images for data augmentation. It utilizes the `icrawler` library to fetch images from Google.

#### Function: `crawl_image_data`
```python
def crawl_image_data(key_word: str, max_num: int, des_dir: str):
    from icrawler.builtin import GoogleImageCrawler
    google_crawler = GoogleImageCrawler(storage={'root_dir': des_dir})
    google_crawler.crawl(keyword=key_word, max_num=max_num)
```

- **Parameters:**
  - `key_word`: The keyword to search for background images.
  - `max_num`: Maximum number of images to download.
  - `des_dir`: Directory to save the downloaded images.

#### Usage Example:
```bash
python crawling.py --keyword 'topview table texture background' --max_num 100 --des_dir ./bg_noises 
```
Also, we'll need paper image for image augmentation

```bash
python crawling.py --keyword 'paper texture background' --max_num 100 --des_dir ./bg_papers 
```

### `augmentation.py`
This script provides functions for augmenting ECG images and labels by adding background noise, applying shadows, and introducing geometric distortions.

#### Global Variables:
- `rectangles`: List to store rectangle coordinates.
- `start_point`, `end_point`: Points for drawing rectangles.
- `drawing`, `delete_mode`: Flags for drawing and deletion modes.
- `image`, `img_copy`, `img_with_text`: Variables for the working image.

#### Key Functions:

##### 1. `add_background_noise`
Adds background noise to an image and adjusts bounding box labels accordingly.

- **Highlights:**
  - Pads the image with random borders.
  - Adds background noise using a provided background and paper image.
  - Adjusts bounding box coordinates in YOLO format.

##### 2. `generate_spot_light_mask`
Generates a spotlight effect mask with Gaussian decay.

- **Parameters:**
  - `mask_size`: Size of the mask.
  - `position`: Center of the spotlight.
  - `max_brightness`, `min_brightness`: Brightness range.
  - `mode`: Decay mode (Gaussian or linear).
  - `speedup`: Optimization flag.

##### 3. `add_spot_light`
Applies the spotlight effect to an image.

- **Parameters:**
  - `image`: Input image.
  - `light_position`: Center of the spotlight.
  - `transparency`: Blend ratio for the effect.

##### 4. `add_bend_distortion`
Applies bend distortion to an image using sinusoidal transformations.

#### Dataset Augmentation Workflow
The script processes images and labels, applying augmentations with background noise and other effects. The augmented data is saved in designated directories.

#### Example Workflow:
```python
if __name__ == "__main__":
    for set in set_list:
        print(f"Current set: {set}")
        image_directory = os.path.join(root, set, "images")
        label_directory = os.path.join(root, set, "labels")
        output_image_directory = os.path.join(root + "_augmented", set, "images")
        output_label_directory = os.path.join(root + "_augmented", set, "labels")

        os.makedirs(output_image_directory, exist_ok=True)
        os.makedirs(output_label_directory, exist_ok=True)

        image_files = [f for f in os.listdir(image_directory) if f.endswith((".jpg", ".png", ".jpeg"))]
        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files), desc="Processing"):
            img = cv2.imread(os.path.join(image_directory, image_file))
            if img is None:
                continue

            mode = random.uniform(0, 1)
            if mode > 0.1:
                bg_files = [f for f in os.listdir(bg_path) if f.endswith((".jpg", ".png", ".jpeg"))]
                pi_files = [f for f in os.listdir(pi_path) if f.endswith((".jpg", ".png", ".jpeg"))]
                bg = cv2.imread(os.path.join(bg_path, random.choice(bg_files)))
                pi = cv2.imread(os.path.join(pi_path, random.choice(pi_files)))
                augimg, boxes = add_background_noise(img, background_img=bg, paper_img=pi, label_file=os.path.join(label_directory, label_file))
                augimg = add_spot_light(augimg)

                cv2.imwrite(os.path.join(output_image_directory, f"{image_file[:-4]}-augmented{image_file[-4:]}"), augimg)
                with open(os.path.join(output_label_directory, f"{label_file[:-4]}-augmented.txt"), 'w') as f:
                    for box in boxes:
                        class_id, x_center, y_center, box_width, box_height = box
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            else:
                cv2.imwrite(os.path.join(output_image_directory, f"{image_file[:-4]}-augmented{image_file[-4:]}"), img)
                shutil.copy(
                    os.path.join(label_directory, label_file),
                    os.path.join(output_label_directory, f"{label_file[:-4]}-augmented.txt")
                )
```

### `roi.py`
This script extracts Regions of Interest (ROIs) from images using a YOLOv7 model and saves them as individual image files along with their bounding box information.

#### Key Function: `detect_and_save_rois`
- Extracts ROIs from input images based on YOLOv7 detections.
- Saves ROIs as separate image files and bounding box information in YOLO format.
- Automatically organizes and sorts the leads based on classes and positions.

#### Usage:
Run the script with the following command:
```bash
python roi.py --weights [pt_file] --source [image_file]
```

#### Arguments:
- `--weights`: Path to the YOLOv7 model file (e.g., `yolov7.pt`).
- `--source`: Path to the image or folder of images for inference.

---

## Notes
- Ensure the `icrawler`, `cv2`, and `torch` libraries are installed before running the scripts.
- The `augmentation.py` script assumes images and labels follow the YOLO format.
- Background images and paper textures should be stored in `bg_noises` and `bg_papers` directories, respectively.
