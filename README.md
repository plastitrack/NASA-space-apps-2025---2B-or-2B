# NASA-space-apps-2025---2B-or-not-2B

Here is the repo for 2B or !2B team, Where the website source code, AI models source code and dataset exists.

# What is the project?

Our solution combines Sentinel-1 SAR satellite data with AI to generate accurate heatmaps of microplastic (MP) distribution in freshwater. SAR detects fine surface variations, while AI analyzes temporal patterns to locate and predict MP concentrations. This provides governments, companies, and communities with an interactive platform for real-time monitoring, identifying pollution hotspots, and guiding water management decisions.By integrating free satellite data, AI modeling, and time-series analysis, the system offers a fast, low-cost, and scalable alternative to traditional sampling—enhancing environmental protection and public awareness.

# Technical Overview

Technologies & Tools: 
Datasets: SLC, GRD from GEE 
Programming: Python 
Algorithm: Biggest Difference
Platform: Google Colab 
Code editor: VS code 
Website: HTML, CSS, JS 
API: FastAPI 

# How to Operate the website?

1. git clone https://github.com/plastitrack/NASA-space-apps-2025---2B-or-not-2B.git
2. cd NASA-space-apps-2025---2B-or-not-2B/backend
3. Run pip install -r requirements.txt
4. python app.py
5. cd ../front-end
6. python -m http.server 5500
7. Go to http://127.0.0.1:5500/
