# 🌋 Heat Flow Analysis in the Panama Basin | CRISP-DM Methodology  
*Statistical modeling of oceanic crust thermal dynamics using Python (Pandas, Statsmodels, Plotly)*  

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python) ![Pandas](https://img.shields.io/badge/Pandas-1.1.2-red) ![Plotly](https://img.shields.io/badge/Plotly-5.0+-lightblue) ![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-orange)

## 📌 Project Overview  
**Goal**: Analyze heat flow data from the Panama Basin to understand its relationship with:  
- Oceanic crust age  
- Geographic coordinates  
- Pollack's (1993) thermal evolution stages  

**Key Achievements**:
- Processed **1,780 samples** from global databases
- Identified **significant correlations** between heat flow and latitude (ρ=-0.45, p<0.001)
- Classified data into **9 thermal stages** following Pollack's model

## 🛠️ Technical Stack
```python
import pandas as pd  # Data cleaning (v1.1.2)
import numpy as np  # Numerical operations (v1.18.5)
import statsmodels.api as sm  # Statistical tests
import plotly.express as px  # Interactive visualizations
from scipy import stats  # Spearman correlation
```

## 📊 Key Visualizations
### 1. Heat Flow Distribution
![Histogram](https://i.imgur.com/heatflow_hist.png)  
Skewness: 11.36 | Kurtosis: 194.91 (Leptokurtic distribution)
### 2. Geospatial Analysis
```python
px.density_mapbox(df, lat='lat', lon='lon', z='Heat_Flow',
                 mapbox_style="carto-darkmatter")
```
![Panama Basin Heat Map](https://i.imgur.com/heatflow_hist.png)  
### 3. Thermal Stage Classification
![Thermal Stages](https://i.imgur.com/heatflow_hist.png)  
Pollack's (1993) model applied to Panama Basin data

## 🔍 Statistical Findings
| Correlation               | Coefficient | p-value  |
|---------------------------|-------------|----------|
| Heat Flow vs Latitude      | -0.45       | <0.001   |
| Heat Flow vs Depth         | 0.16        | <0.001   |
| Latitude vs Depth          | -0.41       | <0.001   |

**Key Insights**:
- Strong negative correlation between heat flow and latitude (p<0.001)
- Weak but significant positive correlation with depth
- Data follows Pollack's (1993) thermal evolution model

## 📂 Repository Structure
```text
/Data
├── HeatFlowIHFC_Panama.txt
├── Pollack_1993_Compilation.csv
/Notebooks
├── CRISP-DM_Analysis.ipynb  # Main workflow
├── GeoVisualization.ipynb    # Interactive maps
/outputs
├── figures/                  # All visualizations
```
## 🚀 How to Replicate
```bash
git clone https://github.com/gacuervol/panama-heatflow.git
pip install -r requirements.txt
jupyter notebook CRISP-DM_Analysis.ipynb
```
## 📜 References
```bibtex
@article{pollack1993heat,
  title={Heat flow from the Earth's interior},
  author={Pollack, Henry N and Hurter, Suzanne J and Johnson, Jeffrey R},
  journal={Reviews of Geophysics},
  volume={31},
  number={3},
  pages={267--280},
  year={1993}
}
```
## 🔗 Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Giovanny_Cuervo-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/tu-perfil)  
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=researchgate)](https://researchgate.net/tu-perfil)  
[![GitHub](https://img.shields.io/badge/GitHub-gacuervol-181717?style=for-the-badge&logo=github)](https://github.com/gacuervol)  
[![Email](https://img.shields.io/badge/Email-giovanny.cuervo%40alu.ulpgc.es-D14836?style=for-the-badge&logo=gmail)](mailto:giovanny.cuervo101@alu.ulpgc.es)

> 💡 **How to reach me**:  
> - **Collaborations**: Open to research partnerships  
> - **Questions**: Best via email  
> - **Job Opportunities**: LinkedIn preferred
