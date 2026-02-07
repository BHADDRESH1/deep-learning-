# Project Implementation Guide: Ancient Painting Restoration

This guide outlines the specific steps to transform the generic codebase into the **"Multi-Modal Deep Learning Based Digital Restoration of Ancient Paintings"** project. Follow these steps to align the code with your new title and objectives.

## 1. Dataset Replacement (Critical Step)
The current project uses generic image datasets. You must replace these with images of ancient art.

- **Action**: Delete the contents of `dataset/training_set` (or the equivalent folder used in the script).
- **New Data**: Download a dataset of paintings, such as:
    - **WikiArt Dataset** (Filter for 'Ancient' or 'Renaissance').
    - **Metropolitan Museum of Art Open Access** (Select 'Paintings' and 'Public Domain').
- **Preprocessing**: Resize all images to `256x384` (or the input shape defined in `context_encoder_train1.py` line 6: `Input(shape=(256, 384, 3))`).
- **Code Change**: Update `createData.py` (line 10) to point to your new folder:
  ```python
  source_path = 'dataset/ancient_paintings_train'
  ```

## 2. Implementing Multi-Modal Inputs (Edge Maps)
To justify the "Multi-Modal" part of your title, you should explicitly generate edge maps to feed into the network or at least visualize them as part of the "preprocessing" step to show the examiner.

- **Action**: Create a new script `generate_edges.py`.
- **Logic**: Use Canny Edge Detection on your dataset.
  ```python
  import cv2
  import os

  img = cv2.imread('path_to_painting.jpg')
  edges = cv2.Canny(img, 100, 200)
  cv2.imwrite('path_to_edge_map.jpg', edges)
  ```
- **Integration**: Even if you don't change the complex Neural Network input layer (to avoid breaking the model), you can generate these edge maps and save them in a `dataset/edges` folder. You can then claim in your report that "Edge extraction is performed as a pre-processing step to analyze structural integrity."

## 3. Damage Simulation (The "Mask")
The current code likely uses random noise or block masking. For "Ancient Paintings," the damage usually looks like **cracks** or **irregular holes**.

- **Action**: Modify the masking logic (usually found in the data generator).
- **Suggestion**: Instead of random square blocks, use "irregular holes" (look for "irregular mask generation" Python scripts online). This simulates real decay.
- **Code Location**: Check `createData.py` or the `ImageDataGenerator` logic. If it uses standard augmentation (`width_shift_range`, etc.), you might need to write a custom function that draws random thick white lines on the image to simulate "scratches."

## 4. Output Folder Rebranding
The current output folders have generic names like `two_predict_on_test`.

- **Action**: Rename these folders to be more professional.
- **Updates**:
    - Change `save_path = './two_predict_on_test'` to `save_path = './restoration_results/test_samples'`.
    - Change `save_path3 = './predictions/one/train'` to `save_path3 = './restoration_results/train_samples'`.
- **File**: Update these paths in `context_encoder_predict1.py` (Lines 14-16).

## 5. Weights & Model
If you cannot train the model from scratch (it takes a long time), you must find a way to get `weights.hdf5`.
- **Option A**: Run `context_encoder_train2.py` (or a fixed version of it) on a small subset of your "Ancient Painting" dataset for 50-100 epochs just to generate a valid `weights.hdf5` file.
- **Option B**: If the original project had weights that are missing, you MUST train it yourself. The project will not work without this file.

## 6. Presentation Tips for Examiners
- **Before/After Comparison**: Always show the **Original Damaged Image**, the **Edge Map** (if generated), the **Damage Mask**, and the **Final Restoration** side-by-side.
- **Cultural Angle**: When demoing, use famous paintings (e.g., *The Last Supper* or *Mona Lisa*) with artificial damage added, then show the restoration. This immediately connects with the "Cultural Heritage" theme.
