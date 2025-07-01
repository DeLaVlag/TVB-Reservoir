
# 🧠 TVB Reservoir

**TVB Reservoir** is a GPU-accelerated reservoir computing framework integrated with **The Virtual Brain (TVB)** platform. 
It enables large-scale reservoir computing with brain network simulations principles as their nodes, 
with CUDA-accelerated kernels for performance-critical tasks such as metric computation.

---

## ✍️ How to Cite

The paper has been accepted for a presentation at the ICANN25 (https://e-nns.org/icann2025/) -34th International Conference on Artificial Neural Networks-
and will be published in its proceedings.

---

## 👤 Author

Michiel van der Vlag

## 🚀 Features

- GPU-accelerated reservoir computing for brain network dynamics  
- Integration with TVB structural connectomes  
- CUDA kernels for key complexity metrics:  
  - **Perturbational Complexity Index (PCI)**  
  - **Detrended Fluctuation Analysis (DFA)**  
  - **Lyapunov Exponent Estimation**  
- Automated output processing and plotting  

---

## ⚡ Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher  
- **CUDA Toolkit** (tested with CUDA 11.x or newer)  
- Python 3.8+  
- Recommended Python packages:  
  - `numpy`  
  - `matplotlib`  
  - `scipy`  
  - `torch` (for regression on GPU)  
  - `pycuda` (for running custom CUDA kernels)  
  - `tvb-library` (for handling TVB connectomes)  
  - `tvb-data` (for importing TVB connectomes)  
  - `tqdm` 
  - `scikit-learn` 

---

## 🛠 Installation

1. Ensure your NVIDIA drivers and CUDA toolkit are properly installed.  
2. Clone this repository:  

   ```bash
   git clone https://github.com/yourusername/tvb-reservoir.git
   cd tvb-reservoir
   ```  
3. Set up a virtual environment (optional but recommended):  

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```  
4. Install dependencies:  

   ```bash
   pip install -r requirements.txt
   ```  

---

## 🧩 Running the Model Drivers

The model drivers are located in the `root/` directory. To run a basic reservoir simulation:

```bash
python model_driver_montbrio.py 
python model_driver_larterbreakspear.py
```

---

## ⚙️ Metrics Kernels

Custom CUDA kernels for computing complexity metrics are located in `metrics/`. The following metrics are supported:

- **PCI**: Estimates system perturbation complexity  
- **DFA**: Assesses long-term temporal correlations  
- **Lyapunov Exponent**: Evaluates sensitivity to initial conditions  

These kernels are automatically invoked during the simulation pipeline when metrics computation is enabled.

To compile kernels manually (if necessary):

## 📊 Plotting Metrics Output

After running a simulation, results are stored in the `data/` directory. To generate the plots for in the ICANN25 paper:

```bash
python plottools/plot_print_all.py 
```

This will produce figures for:  
✔ PCI distribution for each unique TVB simulation
✔ DFA distribution for each unique TVB simulation
✔ Lyapunov distribution for each unique TVB simulation

---

## 📝 Notes

- Requires TVB structural connectivity data  
- CUDA kernels are optimized for single-precision performance  
- Large simulations may require considerable GPU memory  

---


## 🤝 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

**Questions or issues?** Feel free to open an [issue](https://github.com/yourusername/tvb-reservoir/issues) or contact the maintainers.

---