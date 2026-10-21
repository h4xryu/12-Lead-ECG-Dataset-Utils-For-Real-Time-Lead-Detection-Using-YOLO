## Examples

### 1. Lead Detection: Before and After

The following examples illustrate the bounding-box detection results
before and after refinement.

<table>
  <thead>
    <tr>
      <th width="50%">Before</th>
      <th width="50%">After</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img src="images/before1.jpeg" alt="Lead detection before refinement, example 1" width="100%">
      </td>
      <td align="center">
        <img src="images/after1.jpeg" alt="Lead detection after refinement, example 1" width="100%">
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="images/before2.jpeg" alt="Lead detection before refinement, example 2" width="100%">
      </td>
      <td align="center">
        <img src="images/after2.jpeg" alt="Lead detection after refinement, example 2" width="100%">
      </td>
    </tr>
  </tbody>
</table>

### 2. ECG Images and Bounding-Box Annotations

Examples of ECG paper images and their associated lead-region annotations.

<table>
  <thead>
    <tr>
      <th width="50%">ECG Paper Image</th>
      <th width="50%">Bounding-Box Annotations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" valign="middle">
        <img src="images/example1.png" alt="ECG paper image" width="100%">
      </td>
      <td align="center" valign="middle">
        <img src="images/label_example.jpeg" alt="ECG image with bounding-box annotations and class labels" width="100%">
      </td>
    </tr>
  </tbody>
</table>

### 3. Model Training and Evaluation

#### Training Progress

Validation mAP and training loss curves over 150 epochs.

<p align="center">
  <img
    src="images/yolo_training.png"
    alt="YOLO training results showing validation mAP, box loss, class loss, and objectness loss"
    width="100%"
  />
</p>

#### Test-Set Performance by Lead

Test-set average precision at an IoU threshold of 0.5 (AP@0.5) for each ECG lead.

<p align="center">
  <img
    src="images/yolo_ap_by_lead.png"
    alt="Test-set AP at IoU 0.5 for each ECG lead"
    width="100%"
  />
</p>
