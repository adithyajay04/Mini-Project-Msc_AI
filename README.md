# **Multi-Agent Reinforcement Learning Traffic Optimization Benchmarking Suite**

**Academic Mini-Project | Master of Science in Artificial Intelligence**

## **Project Overview**

This repository contains a Multi-Agent Reinforcement Learning (MARL) benchmarking suite developed to evaluate traffic light optimization strategies.  
The system implements a **Multi-Agent Proximal Policy Optimization (MAPPO)** algorithm to dynamically control traffic signals in real-time, replacing traditional fixed-timer systems. The architecture is designed to scale across varying grid complexities, supporting 2x2, 3x3, 4x4, and 5x5 intersection environments to test the viability of decentralized control in traffic management.

### **Key Features**

* **Scalable Architecture:** The PyTorch-based Actor/Critic networks adapt dynamically to the observation spaces of different grid dimensions.  
* **Parallel Execution:** A custom multiprocessing implementation allows multiple Eclipse SUMO simulation instances to run simultaneously for comparative testing.  
* **Control Interface:** A CustomTkinter graphical user interface is provided to launch simulations and toggle between the MAPPO agent and the baseline models.  
* **Performance Tracking:** The system automatically generates Matplotlib graphs to compare the cumulative waiting times of the reinforcement learning agent against the fixed-timer baseline.

## **System Architecture**

* **Physics Engine:** Eclipse SUMO (Simulation of Urban MObility)  
* **Environment Wrapper:** sumo-rl (utilizing the PettingZoo AEC API)  
* **Algorithm:** MAPPO (Multi-Agent PPO) utilizing centralized training and decentralized execution.  
* **Deep Learning Framework:** PyTorch  
* **Data & Interface:** CustomTkinter, Matplotlib, NumPy

### **Repository Structure**

mini-project-msc\_ai/  
│  
├── algorithms/               \# MAPPO algorithm implementation and loss calculations  
├── networks/                 \# PyTorch Actor and Centralized Critic architectures  
├── multi\_layout\_dashboard.py \# Main GUI entry point  
├── run\_instance.py           \# Multiprocessing bridge and simulation execution loop  
├── Train\_Sumo.py             \# Model training script with automated checkpointing  
│  
├── \*.net.xml                 \# SUMO Grid Maps (2x2, 3x3, 4x4, 5x5)  
├── \*.rou.xml                 \# SUMO Route Files mapping vehicle spawn rates  
├── \*.pth                     \# Trained PyTorch Neural Network Weights  
│  
└── requirements.txt          \# Python dependencies

## **Installation & Setup**

**1\. Clone the repository**  
git clone \[https://github.com/\](https://github.com/)\[your-username\]/mini-project-msc\_ai.git  
cd mini-project-msc\_ai

**2\. Install dependencies**  
Ensure Python 3.9 or higher is installed, then run:  
pip install \-r requirements.txt

**3\. Install SUMO**  
The Eclipse SUMO C++ engine is required to run the simulations.

* Download and install from: [Eclipse SUMO](https://eclipse.dev/sumo/)  
* **Configuration:** Ensure the SUMO\_HOME environment variable is correctly set in your system path prior to execution.

## **Usage**

### **Running the Benchmarking Suite**

To evaluate the pre-trained models against the traditional traffic light system, start the control panel:  
python multi\_layout\_dashboard.py

1. Select the simulation mode (MAPPO AI or Fixed Timer Baseline).  
2. Select the desired grid size (e.g., Run 4x4 Grid).  
3. The SUMO simulation will launch and render the traffic environment.  
4. Upon closing the SUMO window, a performance analytics graph will automatically generate and save to the directory.

### **Training a New Model**

To train the neural network on a specific grid configuration, execute the training script with the target map and output paths:  
python Train\_Sumo.py \--net grid4x4.net.xml \--save\_path model\_4x4.pth

*Developed as a Master of Science in Artificial Intelligence Mini-Project by Athul Anilkumar and Adithya Jayaram*
