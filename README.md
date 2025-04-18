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
![Panama Basin Heat Map](https://github.com/gacuervol/oceanic-heatflow/blob/main/figures/PoligonoZona.PNG)  
### 3. Thermal Stage Classification
![Thermal Stages](https://github.com/gacuervol/oceanic-heatflow/blob/main/figures/tabla%20pollack.PNG)
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
├── Oceanic_Heat_Flow_(Pollack_et_al_1993).csv
├── HeatFlowIHFC_Panama.txt
/Notebooks
├── Proyecto_Final (Giovanny A. Cuervo L.).ipynb  # Main workflow
├── GeoVisualization.ipynb    # Interactive maps
/scripts
├── ProjFinal.py  
/figures                  # All visualizations
├── PoligonoZona.PNG
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
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/giovanny-alejandro-cuervo-londo%C3%B1o-b446ab23b/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/Giovanny-Cuervo-Londono)  
[![Email](https://img.shields.io/badge/Email-giovanny.cuervo101%40alu.ulpgc.es-D14836?style=for-the-badge&logo=gmail)](mailto:giovanny.cuervo101@alu.ulpgc.es)  

> 💡 **How to reach me**:  
> - **Collaborations**: Open to research partnerships  
> - **Questions**: Best via email  
> - **Job Opportunities**: LinkedIn preferred
