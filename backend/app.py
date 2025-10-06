from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import ee
import numpy as np
import pandas as pd
import math
import joblib
import warnings
import io
import base64
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import uvicorn
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Initialize FastAPI
app = FastAPI(
    title="Microplastic Prediction API",
    description="API for predicting microplastic concentrations using SAR and oceanographic data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Global variables
MODEL_PATH = 'model_randomforest_ocean.pkl'
model = None

# Pydantic models
class Coordinates(BaseModel):
    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")

class BoundingBox(BaseModel):
    southWest: Coordinates
    northEast: Coordinates

class PredictionRequest(BaseModel):
    bounds: BoundingBox
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    n_tiles: Optional[int] = Field(None, description="Number of tiles (auto-calculated if not provided)")
    spacing_m: Optional[float] = Field(None, description="Spacing between points in meters (auto-calculated if not provided)")
    generate_heatmap: Optional[bool] = Field(True, description="Generate heatmap visualization")
    max_points: Optional[int] = Field(500, description="Maximum number of sample points (for auto-calculation)")
    min_points: Optional[int] = Field(50, description="Minimum number of sample points (for auto-calculation)")

class PredictionResponse(BaseModel):
    status: str
    message: str
    total_points: int
    heatmap_base64: Optional[str] = None
    statistics: Optional[Dict] = None
    risk_distribution: Optional[Dict] = None
    sample_data: Optional[List[Dict]] = None

# Utility Functions
def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project='plastitrack-473720')
        print("✅ Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Earth Engine initialization error: {e}")
        print("   Attempting to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize(project='plastitrack-473720')
            print("✅ Earth Engine initialized after authentication")
            return True
        except Exception as auth_error:
            print(f"❌ Earth Engine authentication failed: {auth_error}")
            return False

def load_model(model_path: str):
    """Load the trained microplastic prediction model"""
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None

def get_oceanographic_data(lon: float, lat: float, date_str: str, buffer_size: int = 2000) -> Dict:
    """Extract mean SST and Chlorophyll-a for a location and date"""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")
        point = ee.Geometry.Point([lon, lat])

        collection = (
            ee.ImageCollection("NASA/OCEANDATA/MODIS-Aqua/L3SMI")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .select(["sst", "chlor_a"])
        )

        if collection.size().getInfo() == 0:
            return {'SST_celsius': np.nan, 'chlorophyll_a': np.nan}

        img = collection.mean()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(buffer_size),
            scale=4000,
            maxPixels=1e9
        ).getInfo()

        # Process SST
        sst_val = stats.get('sst') if stats else None
        if sst_val is None:
            sst_celsius = np.nan
        else:
            s = float(sst_val)
            if s > 1000: s *= 0.01
            if s > 100: s -= 273.15
            sst_celsius = s

        # Process Chlorophyll
        chl_val = stats.get('chlor_a') if stats else None
        chlorophyll_a = float(chl_val) if chl_val is not None else np.nan

        return {'SST_celsius': sst_celsius, 'chlorophyll_a': chlorophyll_a}

    except Exception as e:
        print(f"❌ Ocean data extraction error at ({lon:.4f}, {lat:.4f}): {e}")
        return {'SST_celsius': np.nan, 'chlorophyll_a': np.nan}

def get_sar_image(region, date_str: str):
    """Get Sentinel-1 SAR image for a region and date"""
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .select(["VV", "VH"])
    )

    if collection.size().getInfo() == 0:
        return None

    mean_image = collection.mean()
    mean_image = mean_image.addBands(
        mean_image.select("VV").subtract(mean_image.select("VH")).rename("roughness_proxy")
    )
    return mean_image

def get_sar_data_for_point(sar_image, lon: float, lat: float) -> Dict:
    """Extract SAR data for a specific point"""
    if sar_image is None:
        return {'VV': np.nan, 'VH': np.nan, 'roughness_proxy': np.nan}
    
    try:
        point = ee.Geometry.Point([lon, lat])
        stats = sar_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        return {
            'VV': stats.get('VV', np.nan),
            'VH': stats.get('VH', np.nan),
            'roughness_proxy': stats.get('roughness_proxy', np.nan)
        }
    except Exception as e:
        return {'VV': np.nan, 'VH': np.nan, 'roughness_proxy': np.nan}

def add_extra_features(row_dict: Dict) -> Dict:
    """Add derived geophysical and temporal features"""
    try:
        lon = row_dict.get('lon')
        lat = row_dict.get('lat')
        date_str = row_dict.get('date')
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month = date_obj.month
        day_of_year = date_obj.timetuple().tm_yday

        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        season = 1 if month in [12, 1, 2] else 2 if month in [3, 4, 5] else 3 if month in [6, 7, 8] else 4
        abs_lat = abs(lat)
        
        if lat < -30:
            lat_zone = 0
        elif lat < 0:
            lat_zone = 1
        elif lat < 30:
            lat_zone = 2
        else:
            lat_zone = 3

        VV = row_dict.get('vv_backscatter')
        VH = row_dict.get('vh_backscatter')
        SST = row_dict.get('sst')
        chl = row_dict.get('chlorophyll')
        roughness_proxy = row_dict.get('roughness_proxy')

        cross_pol_ratio = (VH / VV) if (VV not in [None, 0] and VH is not None and not np.isnan(VV) and not np.isnan(VH)) else np.nan
        VV_VH_ratio = VV / VH if (VH not in [None, 0] and VH is not None and not np.isnan(VV) and not np.isnan(VH)) else np.nan
        backscatter_diff = VV - VH if (VV is not None and VH is not None and not np.isnan(VV) and not np.isnan(VH)) else np.nan
        backscatter_sum = VV + VH if (VV is not None and VH is not None and not np.isnan(VV) and not np.isnan(VH)) else np.nan

        row_dict.update({
            'month_sin': month_sin,
            'month_cos': month_cos,
            'season': season,
            'day_of_year': day_of_year,
            'abs_lat': abs_lat,
            'lat_zone': lat_zone,
            'cross_pol_ratio': cross_pol_ratio,
            'VV_VH_ratio': VV_VH_ratio,
            'backscatter_diff': backscatter_diff,
            'backscatter_sum': backscatter_sum,
            'VV_backscatter': VV,
            'VH_backscatter': VH,
            'SST_celsius': SST,
            'chlorophyll_a': chl,
            'VV_SST_interaction': VV * SST if (VV is not None and SST is not None and not np.isnan(VV) and not np.isnan(SST)) else np.nan,
            'VH_SST_interaction': VH * SST if (VH is not None and SST is not None and not np.isnan(VH) and not np.isnan(SST)) else np.nan,
            'VH_chl_interaction': VH * chl if (VH is not None and chl is not None and not np.isnan(VH) and not np.isnan(chl)) else np.nan,
            'roughness_SST': roughness_proxy * SST if (roughness_proxy is not None and SST is not None and not np.isnan(roughness_proxy) and not np.isnan(SST)) else np.nan,
            'sst_squared': SST ** 2 if (SST is not None and not np.isnan(SST)) else np.nan,
            'chl_squared': chl ** 2 if (chl is not None and not np.isnan(chl)) else np.nan,
            'log_chlorophyll': math.log1p(chl) if (chl is not None and not np.isnan(chl) and chl >= 0) else np.nan
        })
    except Exception as e:
        print(f"⚠️ Feature addition error: {e}")
    return row_dict

def predict_microplastic(data_dict: Dict, model) -> float:
    """Make prediction using the trained model"""
    if model is None:
        return np.nan
    
    try:
        feature_order = [
            'VV_backscatter', 'roughness_proxy', 'cross_pol_ratio',
            'month_cos', 'season', 'day_of_year', 'abs_lat', 'lat_zone',
            'VV_SST_interaction', 'VH_SST_interaction', 'VH_chl_interaction',
            'chl_squared', 'backscatter_diff', 'backscatter_sum', 'roughness_SST'
        ]
        
        feature_values = [data_dict.get(feat, np.nan) for feat in feature_order]
        X_pred = pd.DataFrame([feature_values], columns=feature_order)
        prediction = model.predict(X_pred)[0]
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return np.nan

def create_tiles(sw_lon: float, sw_lat: float, ne_lon: float, ne_lat: float, n_tiles: int = 25) -> List[Polygon]:
    """Divide bounding box into tiles"""
    grid_size = int(np.sqrt(n_tiles))
    if grid_size ** 2 != n_tiles:
        raise ValueError(f"n_tiles must be a perfect square, got {n_tiles}")
    
    dx = (ne_lon - sw_lon) / grid_size
    dy = (ne_lat - sw_lat) / grid_size
    
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile_coords = [
                (sw_lon + j*dx, sw_lat + i*dy),
                (sw_lon + (j+1)*dx, sw_lat + i*dy),
                (sw_lon + (j+1)*dx, sw_lat + (i+1)*dy),
                (sw_lon + j*dx, sw_lat + (i+1)*dy)
            ]
            tiles.append(Polygon(tile_coords))
    
    return tiles

def generate_sample_points(tile: Polygon, spacing_m: float = 100) -> List[tuple]:
    """Generate sample points within a tile"""
    minx, miny, maxx, maxy = tile.bounds
    
    # Use latitude-aware spacing calculation
    lat_center = (miny + maxy) / 2
    spacing_deg_lat = spacing_m / 111000  # ~111km per degree latitude
    spacing_deg_lon = spacing_m / (111000 * np.cos(np.radians(lat_center)))  # Adjust for longitude
    
    lons = np.arange(minx, maxx, spacing_deg_lon)
    lats = np.arange(miny, maxy, spacing_deg_lat)
    
    # Ensure at least the center point if no grid points
    if len(lons) == 0:
        lons = np.array([(minx + maxx) / 2])
    if len(lats) == 0:
        lats = np.array([(miny + maxy) / 2])
    
    points = []
    for lon in lons:
        for lat in lats:
            p = Point(lon, lat)
            if tile.contains(p) or tile.touches(p):
                points.append((lon, lat))
    
    # If still no points, add center point
    if len(points) == 0:
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        points.append((center_x, center_y))
    
    return points

def calculate_optimal_parameters(sw_lat: float, sw_lng: float, ne_lat: float, ne_lng: float, 
                                 max_points: int = 500, min_points: int = 50) -> tuple:
    """
    Automatically calculate optimal n_tiles and spacing_m based on area size.
    
    Returns:
        tuple: (n_tiles, spacing_m, estimated_points)
    """
    # Calculate area dimensions in kilometers
    lat_diff = abs(ne_lat - sw_lat)
    lng_diff = abs(ne_lng - sw_lng)
    
    # Average latitude for more accurate longitude distance
    avg_lat = (sw_lat + ne_lat) / 2
    
    # Distance calculations (approximate)
    lat_km = lat_diff * 111.0  # 1 degree latitude ≈ 111 km
    lng_km = lng_diff * 111.0 * np.cos(np.radians(avg_lat))  # Adjust for latitude
    
    area_km2 = lat_km * lng_km
    
    print(f"📏 Area dimensions: {lat_km:.2f} km × {lng_km:.2f} km = {area_km2:.2f} km²")
    
    # Determine appropriate resolution based on area size
    if area_km2 < 1:  # Very small area (< 1 km²)
        n_tiles = 4
        spacing_m = 100
    elif area_km2 < 10:  # Small area (1-10 km²)
        n_tiles = 9
        spacing_m = 200
    elif area_km2 < 100:  # Medium area (10-100 km²)
        n_tiles = 16
        spacing_m = 500
    elif area_km2 < 1000:  # Large area (100-1000 km²)
        n_tiles = 25
        spacing_m = 1000
    elif area_km2 < 10000:  # Very large area (1000-10000 km²)
        n_tiles = 36
        spacing_m = 2000
    else:  # Huge area (> 10000 km²)
        n_tiles = 49
        spacing_m = 3000
    
    # Estimate total points
    grid_size = int(np.sqrt(n_tiles))
    tile_width_km = lng_km / grid_size
    tile_height_km = lat_km / grid_size
    
    points_per_tile_x = max(1, int(tile_width_km * 1000 / spacing_m))
    points_per_tile_y = max(1, int(tile_height_km * 1000 / spacing_m))
    estimated_points = n_tiles * points_per_tile_x * points_per_tile_y
    
    # Adjust if estimated points exceed limits
    if estimated_points > max_points:
        # Increase spacing to reduce points
        scaling_factor = np.sqrt(estimated_points / max_points)
        spacing_m = int(spacing_m * scaling_factor)
        estimated_points = int(estimated_points / (scaling_factor ** 2))
        print(f"⚠️ Adjusted spacing to {spacing_m}m to limit points to ~{max_points}")
    
    if estimated_points < min_points and area_km2 > 0.1:
        # Decrease spacing to increase points
        scaling_factor = np.sqrt(min_points / estimated_points)
        spacing_m = max(50, int(spacing_m / scaling_factor))
        estimated_points = int(estimated_points * (scaling_factor ** 2))
        print(f"⚠️ Adjusted spacing to {spacing_m}m to reach ~{min_points} points")
    
    print(f"🎯 Optimal parameters: {n_tiles} tiles, {spacing_m}m spacing → ~{estimated_points} points")
    
    return n_tiles, spacing_m, estimated_points
    """Generate sample points within a tile"""
    minx, miny, maxx, maxy = tile.bounds
    
    # Use latitude-aware spacing calculation
    lat_center = (miny + maxy) / 2
    spacing_deg_lat = spacing_m / 111000  # ~111km per degree latitude
    spacing_deg_lon = spacing_m / (111000 * np.cos(np.radians(lat_center)))  # Adjust for longitude
    
    lons = np.arange(minx, maxx, spacing_deg_lon)
    lats = np.arange(miny, maxy, spacing_deg_lat)
    
    # Ensure at least the center point if no grid points
    if len(lons) == 0:
        lons = np.array([(minx + maxx) / 2])
    if len(lats) == 0:
        lats = np.array([(miny + maxy) / 2])
    
    points = []
    for lon in lons:
        for lat in lats:
            p = Point(lon, lat)
            if tile.contains(p) or tile.touches(p):
                points.append((lon, lat))
    
    # If still no points, add center point
    if len(points) == 0:
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        points.append((center_x, center_y))
    
    return points

def generate_heatmap(df: pd.DataFrame) -> str:
    """Generate heatmap with transparent background, return as base64 encoded image"""
    try:
        lons = df['lon'].values
        lats = df['lat'].values
        concentrations = df['predicted_mp_concentration'].dropna().values
        
        if len(concentrations) == 0:
            return None
        
        # Filter to valid predictions
        valid_mask = ~np.isnan(df['predicted_mp_concentration'])
        lons = df.loc[valid_mask, 'lon'].values
        lats = df.loc[valid_mask, 'lat'].values
        concentrations = df.loc[valid_mask, 'predicted_mp_concentration'].values
        
        # Create interpolated grid
        grid_resolution = 200
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        lon_padding = (lon_max - lon_min) * 0.02
        lat_padding = (lat_max - lat_min) * 0.02
        
        grid_lon = np.linspace(lon_min - lon_padding, lon_max + lon_padding, grid_resolution)
        grid_lat = np.linspace(lat_min - lat_padding, lat_max + lat_padding, grid_resolution)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        grid_concentration = griddata(
            points=(lons, lats),
            values=concentrations,
            xi=(grid_lon_mesh, grid_lat_mesh),
            method='cubic',
            fill_value=np.nan
        )
        
        # Create heatmap with transparent background
        colors = ['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', 
                  '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
        cmap = LinearSegmentedColormap.from_list('microplastic', colors, N=100)
        
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
        
        # Make figure background transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Create contour plot
        contour = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_concentration, 
                              levels=50, cmap=cmap, extend='max', alpha=0.85)
        
        # Add subtle contour lines
        ax.contour(grid_lon_mesh, grid_lat_mesh, grid_concentration,
                   levels=10, colors='white', linewidths=0.5, alpha=0.4)
        
        # Plot data points
        ax.scatter(lons, lats, c=concentrations, cmap=cmap,
                  s=25, edgecolors='white', linewidths=0.8, alpha=0.9, zorder=5)
        
        # Colorbar with transparent background
        cbar = plt.colorbar(contour, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Microplastic Concentration (particles/L)', 
                       fontsize=14, fontweight='bold', color='white')
        cbar.ax.tick_params(labelsize=11, colors='white')
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(2)
        
        # Add risk level reference lines on colorbar
        risk_levels = [0.1, 0.5, 1.0]
        risk_labels = ['Low', 'Moderate', 'High']
        for level, label in zip(risk_levels, risk_labels):
            if level <= concentrations.max():
                cbar.ax.axhline(y=level, color='white', linestyle='--', linewidth=2, alpha=0.7)
                cbar.ax.text(1.8, level, label, fontsize=10, color='white', 
                            fontweight='bold', va='center')
        
        # Style axes
        ax.set_xlabel('Longitude', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold', color='white')
        ax.set_title('Microplastic Concentration Heatmap',
                     fontsize=18, fontweight='bold', pad=20, color='white')
        
        # White grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='white')
        
        # White tick labels
        ax.tick_params(colors='white', labelsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.5)
        
        # Add statistics box with semi-transparent background
        stats_text = (f"Total Points: {len(df.loc[valid_mask])}\n"
                      f"Mean: {concentrations.mean():.3f} particles/L\n"
                      f"Max: {concentrations.max():.3f} particles/L\n"
                      f"Min: {concentrations.min():.3f} particles/L")
        
        props = dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white', linewidth=2)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace', color='white')
        
        plt.tight_layout()
        
        # Convert to base64 with transparency
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   transparent=True, facecolor='none', edgecolor='none')
        buf.seek(0)
        heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return heatmap_base64
        
    except Exception as e:
        print(f"❌ Heatmap generation error: {e}")
        return None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        print("⚠️ Warning: Earth Engine not initialized. API will have limited functionality.")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Microplastic Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Generate microplastic predictions for a region",
            "GET /health": "Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "earth_engine": "initialized" if ee.data._credentials else "not initialized",
        "model": "loaded" if model is not None else "not loaded"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_microplastics(request: PredictionRequest):
    """
    Generate microplastic predictions for a given bounding box
    
    Parameters:
    - bounds: Bounding box with southWest and northEast coordinates
    - date: Date in YYYY-MM-DD format (defaults to today)
    - n_tiles: Number of tiles to divide region into (must be perfect square)
    - spacing_m: Spacing between sample points in meters
    - generate_heatmap: Whether to generate visualization
    
    Returns:
    - CSV data with predictions
    - Base64 encoded heatmap images (optional)
    - Statistics and risk distribution
    """
    
    try:
        # Validate and prepare inputs
        sw_lat = request.bounds.southWest.lat
        sw_lng = request.bounds.southWest.lng
        ne_lat = request.bounds.northEast.lat
        ne_lng = request.bounds.northEast.lng
        
        # Use current date if not provided
        date_str = request.date if request.date else datetime.now().strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Auto-calculate n_tiles and spacing_m if not provided
        if request.n_tiles is None or request.spacing_m is None:
            print("🤖 Auto-calculating optimal parameters...")
            n_tiles, spacing_m, estimated_points = calculate_optimal_parameters(
                sw_lat, sw_lng, ne_lat, ne_lng, 
                max_points=request.max_points,
                min_points=request.min_points
            )
        else:
            n_tiles = request.n_tiles
            spacing_m = request.spacing_m
            
            # Validate n_tiles is perfect square
            grid_size = int(np.sqrt(n_tiles))
            if grid_size ** 2 != n_tiles:
                raise HTTPException(status_code=400, detail=f"n_tiles must be a perfect square (e.g., 4, 9, 16, 25, 36)")
            
            estimated_points = "unknown"
        
        print(f"🌊 Processing request for bounds: ({sw_lat}, {sw_lng}) to ({ne_lat}, {ne_lng})")
        print(f"📅 Date: {date_str}")
        print(f"🔲 Tiles: {n_tiles}, Spacing: {spacing_m}m, Estimated points: {estimated_points}")
        
        # Create tiles
        tiles = create_tiles(sw_lng, sw_lat, ne_lng, ne_lat, n_tiles)
        
        # Create region for SAR
        polygon_coords = [
            (sw_lng, sw_lat),
            (ne_lng, sw_lat),
            (ne_lng, ne_lat),
            (sw_lng, ne_lat),
            (sw_lng, sw_lat)
        ]
        ee_region = ee.Geometry.Polygon([polygon_coords])
        
        # Get SAR image
        print("🛰️ Fetching Sentinel-1 SAR data...")
        sar_image = get_sar_image(ee_region, date_str)
        
        all_rows = []
        
        for tile_idx, tile in enumerate(tiles):
            sample_points = generate_sample_points(tile, spacing_m)
            
            print(f"🔲 Processing Tile {tile_idx + 1}/{len(tiles)}: {len(sample_points)} points")
            
            if len(sample_points) == 0:
                continue
            
            # Get oceanographic data for tile
            lon, lat = sample_points[0]
            ocean_data = get_oceanographic_data(lon, lat, date_str)
            
            print(f"   Ocean data - SST: {ocean_data['SST_celsius']}, Chl: {ocean_data['chlorophyll_a']}")
            
            # Process each point
            for point_lon, point_lat in sample_points:
                sar_data = get_sar_data_for_point(sar_image, point_lon, point_lat)
                
                row = {
                    'lon': point_lon,
                    'lat': point_lat,
                    'date': date_str,
                    'vv_backscatter': sar_data['VV'],
                    'vh_backscatter': sar_data['VH'],
                    'sst': ocean_data['SST_celsius'],
                    'chlorophyll': ocean_data['chlorophyll_a'],
                    'roughness_proxy': sar_data['roughness_proxy']
                }
                
                row = add_extra_features(row)
                
                if model is not None:
                    prediction = predict_microplastic(row, model)
                    row['predicted_mp_concentration'] = prediction
                else:
                    row['predicted_mp_concentration'] = np.nan
                
                all_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No valid data points found in the specified region")
        
        # Calculate statistics
        valid_predictions = df['predicted_mp_concentration'].dropna()
        
        statistics = None
        risk_distribution = None
        
        if len(valid_predictions) > 0:
            statistics = {
                "total_points": len(df),
                "valid_predictions": len(valid_predictions),
                "min": float(valid_predictions.min()),
                "max": float(valid_predictions.max()),
                "mean": float(valid_predictions.mean()),
                "median": float(valid_predictions.median()),
                "std": float(valid_predictions.std())
            }
            
            high_risk = (valid_predictions >= 1.0).sum()
            moderate_risk = ((valid_predictions >= 0.5) & (valid_predictions < 1.0)).sum()
            low_risk = ((valid_predictions >= 0.1) & (valid_predictions < 0.5)).sum()
            very_low_risk = (valid_predictions < 0.1).sum()
            
            risk_distribution = {
                "high_risk": {"count": int(high_risk), "percentage": float(100*high_risk/len(valid_predictions))},
                "moderate_risk": {"count": int(moderate_risk), "percentage": float(100*moderate_risk/len(valid_predictions))},
                "low_risk": {"count": int(low_risk), "percentage": float(100*low_risk/len(valid_predictions))},
                "very_low_risk": {"count": int(very_low_risk), "percentage": float(100*very_low_risk/len(valid_predictions))}
            }
        
        # Generate visualizations
        heatmap_base64 = None
        
        if request.generate_heatmap and len(valid_predictions) > 0:
            print("🎨 Generating visualization...")
            heatmap_base64 = generate_heatmap(df)
    
       
        return PredictionResponse(
            status="success",
            message=f"Successfully processed {len(df)} points",
            total_points=len(df),
            heatmap_base64=heatmap_base64,
            statistics=statistics,
            risk_distribution=risk_distribution,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)