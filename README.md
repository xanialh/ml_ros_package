# Machine Learning Pipeline for Social Density Estimation on Grid Maps

A Final Year BSc Computer Science Graduation Project completed at Cardiff University (2024).

**Author:** Daniel Hixson  
**Project Supervisor:** Steven Silva Mendoza  
**Moderator:** Carolina Fuentes Toro  

---

## Project Overview

In environments where autonomous robots coexist with humans (such as smart warehouses, hospitals, and homes), navigation safety and efficiency present a substantial challenge. Traditional robotic navigation frameworks rely on static mapping or purely reactive path adjustments, which frequently struggle to adapt to dynamic, unpredictable human motion. 

This project implements a modular **Machine Learning Pipeline** integrated into the **Robot Operating System (ROS)** to estimate and predict social density grid maps. By training deep learning models to analyze environmental occupancy grids, the system classifies spatial cells into **Low**, **Medium**, and **High** social density areas. This predictive capability supplies crucial structural context to a robot's path planner, allowing it to proactively generate paths that minimize human disruptions and maximize navigation safety.

---

## Project Justification & Objectives

This work builds upon the social navigation framework introduced by Silva et al., which tracks dynamic human "agents" in ROS and models their personal comfort boundaries as circular "social heatmaps". 

The explicit goal of this project is to bridge this framework with learning-based methods. By utilizing deep neural networks to identify patterns where human congestion is likely to occur, a sampling-based path planner can actively avoid historically high-density social zones altogether, rather than merely reacting when a collision is imminent.

---

## Pipeline Architecture

The project is engineered as a clean, linear, and modular pipeline where each independent stage handles a specific aspect of data generation, management, or execution:

1. **Training Route Simulation (`training_route.py`):**
   * Automatically commands a simulated robot to navigate through a sequence of waypoints in a crowded social environment.
   * Uses an automated bash script (`route_script.sh`) to loop simulation routes continuously to accumulate a robust dataset.
2. **Data Collection (`record_maps.py` & `record_maps_coords.py`):**
   * Acts as a live ROS subscriber capturing spatial data streams from active topics.
   * Formats and logs static obstacle grid maps, dynamic agent-driven social heat maps, and robot positional odometry into clean text files.
3. **Data Splitting (`split_data.py`):**
   * Segregates raw recorded outputs into distinct inputs (obstacle grid maps/coordinates) and ground-truth labels (social heat maps).
4. **Data Preprocessing & Augmentation:**
   * Collapses chronological map segments into consolidated representations, applies data normalization, and utilizes cropping, padding, and resizing to structure uniform tensors for deep learning model ingestion.
5. **Model Training:**
   * Coordinates the training of neural network architectures built with PyTorch, iteratively adjusting weights and biases to improve spatial classification performance.
6. **Evaluation (`evaluation.py`):**
   * Subjects trained models to unseen validation data across diverse metrics to determine operational limits and generalizability.
7. **Model Deployment:**
   * Wraps the finalized predictive model into a live ROS node equipped with a subscriber to interpret incoming obstacle grids and a publisher to broadcast real-time social density maps.

---

## Simulation Environment & Assets

Data generation and virtual validation are managed via a highly structured simulation environment:
* **The Robot Platform:** A simulated **Pepper Robot** (SoftBank Robotics), a human-centric interactive platform ideally suited for public and social spaces.
* **Simulation Software:** Powered by **Gazebo** for physics-based environmental modeling and **RViz** for real-time visualization of paths, coordinates, and active grid structures.
* **Environment Maps:** To test operational versatility, data was harvested across four structurally distinct simulated environments: *Bookstore*, *Office*, *Small House*, and *Small Warehouse*.
* **Social Agents:** Human actors are emulated utilizing a social motion model. The agents behave dynamically, showing discomfort and moving away when crowded, and are capable of traversing environments alone or in coupled groups.

---

## Deep Learning Models & Architecture

Two distinct deep learning approaches were developed using **PyTorch** to evaluate their effectiveness at handling grid-map classification as a semantic segmentation problem:

### 1. Fully Convolutional Networks (FCN)
* **Design Philosophy:** Tailored specifically for dense pixel-wise semantic segmentation tasks.
* **Mechanics:** Replaces fully connected layers entirely with convolutional operations, maintaining full structural spatial resolution from input to output.
* **Constraint:** Restricted exclusively to spatial map arrays; it cannot naturally ingest auxiliary numerical metadata like exact robot coordinates.

### 2. Convolutional Neural Networks (CNN)
* **Design Philosophy:** Structured to allow multi-modal feature fusion.
* **Mechanics:** Integrates traditional convolutional and pooling structures with flexible linear layers.
* **Advantage:** Capable of processing two-dimensional spatial obstacle maps in tandem with non-spatial numerical vectors, allowing the model to incorporate the robot's current coordinates to better map spatial features.

---

## Results, Limitations, and Future Work

### Key Performance Insights
* **Isolating Low Activity:** Both networks successfully learned to identify and segment areas characterizing low social activity across all tested maps.
* **The Impact of Class Imbalance:** The unweighted variations of both models struggled to effectively classify medium and high social density zones. This behavior stems directly from a heavily skewed training dataset, where open, low-activity cells overwhelmingly dominate the environment.
* **Coordinate Integration:** Fusing non-spatial robot coordinates into the CNN architecture measurably assisted the model in learning location-specific spatial structures. However, applying rigid weights to the Cross-Entropy loss function did not completely overcome the data scarcity for high-congestion zones.

### Project Limitations
* **Simulation Bottlenecks:** Data collection efficiency was limited because the simulation could not be artificially accelerated. Additionally, long-distance waypoint tracking occasionally encountered edge-case freezes where the robot became stuck on geometry or frozen near a social agent, requiring manual simulation resets.
* **Validation Split:** The final implementation did not dynamically utilize a dedicated validation subset during the training loops to automate hyperparameter tuning.

### Future Opportunities
* **Advanced Loss Adjustments:** Transitioning from Cross-Entropy loss to **Focal Loss** or integrating cost-sensitive learning principles to disproportionately penalize errors on rare, high-density cells.
* **Dataset Balancing:** Leveraging targeted oversampling of high-density scenarios or careful undersampling of low-activity spaces to create balanced class training sets.
* **Percentile Class Boundaries:** Altering data labeling by defining low, medium, and high social classes dynamically based on statistical percentiles rather than hard-coded numeric thirds.
* **Feature Engineering Expansion:** Expanding network inputs to process additional vital environment context metrics, such as real-time proximity to Points of Interest (POIs), agent group sizes, and instantaneous robot velocity vectors.

---

## Technical Stack & Dependencies

* **Operating System:** Linux
* **Development IDE:** Visual Studio Code (VS Code)
* **Robotics Suite:** ROS Noetic (ROS 1 version 1.16.0)
* **Simulation Core:** Gazebo & RViz
* **Language:** Python 3.8.10
* **Deep Learning Engine:** PyTorch 2.2.2
* **Version Control:** Git & GitHub
