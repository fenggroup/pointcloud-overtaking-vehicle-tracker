# **Quantifying Drivers-Overtaking-Bicyclists with Surrogate Safety Measures Derived from a High-Resolution Digital Lidar**

This repository contains a modular Python pipeline for analyzing bicycle–vehicle interactions using LiDAR point-cloud data.

The algorithm detects vehicles, tracks them across frames, estimates bicycle and vehicle speeds, summarizes passing events, and computes surrogate safety metrics.

## How to run

> **Note:** Ensure the data submodule is initialized before running the pipeline (see *Data Access* below).

## Data Access (Required)

This repository uses a **Git submodule** to manage the LiDAR dataset required to run the pipeline.

1. ### Clone this repository

- #### Clone with data (recommended)
    ```bash
    git clone --recurse-submodules https://github.com/fenggroup/pointcloud-overtaking-vehicle-tracker.git
    cd pointcloud-overtaking-vehicle-tracker
    ```
- #### If you already cloned without data
    ```bash
    git submodule update --init --recursive
    ```
- #### Expected data location
    
    After initialization, the dataset will be available at:
    
    ```bash
    data/pointclouds/frame-XXXXXX.pcd
    ```
**The pipeline will not run correctly unless the data submodule is initialized.**



2. ### Set up a virtual environment
```
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. ### Install dependencies
```
python -m pip install -r requirements.txt
```

4. ### Run the full code
```
python -m src.main
```

**You will be prompted whether to recompute bike speeds if they already exist. If already computed and the file already exists no need to run it again so you can simply reply by 'N'.**


### Dependencies

See ```requirements.txt```.

This code was tested with Python 3.10-3.12.

Key dependencies include:

- **numpy < 2.0**

- **open3d**

- **scikit-learn**

## Project Structure

```text
project_root/
├── data/           #Git submodule (dataset repository)
│   └── pointclouds/
│       └── frame-XXXXXX.pcd
│
├── outputs/
│   ├── bike-speed-per-frame/
│   │   └──bike-speed-estimation.csv
│   │
│   │── detection-and-tracking/
│   │   ├──vehicle-tracking-frames_raw.csv
│   │   └──vehicle-tracking-frames-smoothed.csv
│   │
│   ├── passing-events/
│   │   └──passing-events-summary.csv
│   │      
│   ├── vehicles-speed/
│   │   └──vehicle-speeds-frames.csv
│   │ 
│   └── metrics/
│       └──surrogate-safety-metrics.csv
│
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── main.py
    │
    ├── speeds_computation/
    │   ├── __init__.py
    │   ├── bike_speed_estimation.py
    │   └── vehicle_speed_per_frame.py
    │
    ├── detection_tracking/
    │   ├── __init__.py
    │   ├── detection.py
    │   ├── tracking.py
    │   ├── box_fitting.py
    │   └── vehicle_tracking.py
    │
    ├── extracting_passing_events/
    │   ├── __init__.py
    │   └── passing_events.py
    │
    ├── smoothing/
    │   ├── __init__.py
    │   └── smoothing.py
    │
    ├── metrics/
    │   ├── __init__.py
    │   └── surrogate_safety_metrics.py
    │
    ├── pcd_io.py
    └── ground_segmentation.py
```

## Algorithm Overview

The algorithm executes the following steps **in order**:

1. **Bike speed estimation** (ICP-based)
2. **Vehicle detection and tracking**
3. **Passing event summary**
4. **Temporal smoothing**
5. **Vehicle speed computation**
6. **Surrogate safety metric computation**

All steps are orchestrated through a single entry point:

`python -m src.main`

## Code Modules (What Each File Does)


### 1. High-level pipeline runner

`src/main.py`

- Optionally runs bike speed estimation (if output does not exist or user chooses to recompute)

- Executes all downstream processing steps in sequence

### 2. Point cloud I/O
`pcd_io.py`

- Handles reading point cloud frames from .pcd files using Open3D.

<!-- 
- **Functions**:

    - load_o3d_from_pcd(path): Loads a point cloud from the given PCD file path using Open3D.

    - convert_to_numpy(pcd): Converts an Open3D point cloud into a NumPy N×3 array of (x, y, z) points. 


- **Used in**:

    - All major modules that read PCD frames (e.g., bike_speed_estimation.py, vehicle_tracking.py, etc.)

- **Purpose**:

    - Centralizes PCD file reading logic.

    - Enables interoperability between Open3D and NumPy.
 -->
 
### 3. Ground segmentation/filtering
`ground_segmentation.py`

- Segments ground points from point cloud frames using a custom RANSAC-based method.
- Improves detection quality by isolating objects from road surface.
- Makes clustering faster and more accurate.
  
<!--
- **Functions**:

    - segment_ground(points, dist_thresh, angle_thresh, max_attempts):
    - Returns two arrays: ground_points and non_ground_points based on a plane-fitting strategy with distance & angle constraints.


- **Used in**:

    - vehicle_tracking.py to remove ground-level noise before DBSCAN clustering.
-->



 

### 4. Speed Estimation
`bike_speed_estimation.py`

- Estimates bicycle speed using frame-to-frame LiDAR registration

- Uses FPFH + RANSAC + ICP

- Performs outlier removal, interpolation, and Savitzky–Golay smoothing


**Output**:
- `outputs/bike-speed-icp.csv`



### 5. Vehicle Detection & Tracking

5.1.  `detection.py`
- Loads point clouds

- Applies lateral ROI filtering

- Performs ground segmentation

- Clusters objects using DBSCAN



5.2.  `tracking.py`

- Assigns persistent vehicle IDs across frames

- Filters objects based on longitudinal motion consistency



5.3.  `box_fitting.py`


- Fits hugged axis-aligned bounding boxes to vehicles

- Detects passing start/end using front/rear thresholds

- Filters out short-lived tracks




5.4.  `vehicle_tracking.py`

- Orchestrates detection, tracking, and box fitting

- Displays a progress bar only

- Outputs per-frame vehicle tracking data


**Output**:

- `outputs/vehicle-tracking-frames-raw.csv`


### 6. Vehicle Speed Computation
`vehicle_speed_per_frame.py`

**Computes**:

- Relative speed (vehicle vs. bicycle)

- Absolute vehicle speed

- Applies temporal smoothing to speed profiles

- Operates only on passing vehicles

**Output**:

- `results/vehicle_speeds_frames.csv`


## 7. Smoothing

`smoothing.py`

- Applies temporal smoothing to:

  - Vehicle trajectories

  - Bicycle speed (if required downstream)

**Output:**

`outputs/vehicle-tracking-frames-smoothed.csv`






### 8. Surrogate Safety Metrics
`surrogate_safety_metrics.py`

**Computes**:

- Passing distance

- Passing speed

- Passing duration 


**Output**:

- `results/surrogate_safety_metrics.csv`


## Output CSV Files & Column Descriptions

1. `bike-speed-estimation.csv`
   
| Column                | Description                         |
|-----------------------|-------------------------------------|
|start_frame            |Lidar frame i to be processed        |
|end_frame              |Lidar frame i+1 to be processed      |
|bike_displacement_m    |Frame-to-frame displacement (meters) |
|bike_speed_mph	        |Raw bike speed (mph)                 |
|bike_speed_smoothed_mph|Smoothed bike speed (mph)            |



2. `vehicle-tracking-frames-raw.csv`
   
| Column                | Description                         |
|-----------------------|-------------------------------------|
|frame	                |LiDAR frame index|
|vehicle_id	            |Persistent vehicle identifier|
|centroid_x/y/z	        |Vehicle centroid coordinates|
|min_distance_m	        |Closest point distance to bike|
|box_center_x/y/z       |Bounding box center coordinates|
|box_length_m	        |Vehicle length|
|box_width_m	        |Vehicle width|
|box_height_m	        |Vehicle height|
|front_x	            |Front face x-coordinate|
|rear_x	                |Rear face x-coordinate|
|is_passing_frame	    |Binary indicator of passing|


3. `vehicle-speeds-frames.csv`

| Column                        | Description                         |
|-------------------------------|-------------------------------------|
|relative_speed_mph         	|Vehicle speed relative to bike|
|vehicle_speed_mph	            |Absolute vehicle speed|
|relative_speed_smoothed_mph 	|Smoothed relative speed|
|vehicle_speed_smoothed_mph	    |Smoothed absolute speed|



4. `passing-events-summary.csv`
   
| Column            | Description                         |
| ------------------|-------------------------------------|
|vehicle_id	        |Persistent identifier of the tracked vehicle|
|track_start_frame	|First LiDAR frame in which the vehicle track appears|
|track_end_frame	|Last LiDAR frame in which the vehicle track appears|
|pass_start_frame	|Frame index where the passing maneuver begins|
|pass_end_frame	    |Frame index where the passing maneuver ends|
|passing_frame	    |Frame index of passing distance (closest approach)|
|passing_duration_s	|Duration of the passing maneuver (seconds)|
|passing_distance_m	|Minimum lateral distance between vehicle and bicycle (meters)|
|status	            |Event label indicating if a vehicle is indeed passing, partially passing or not passing|





- **pypcd4**

- **pandas**

- **scipy**
