// Initialize the map
const map = L.map('map').setView([51.505, -0.09], 13);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19
}).addTo(map);

// Initialize the draw control
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const drawControl = new L.Control.Draw({
    draw: {
        rectangle: {
            shapeOptions: {
                color: '#667eea',
                weight: 3,
                fillOpacity: 0.3
            }
        },
        polygon: false,
        circle: false,
        marker: false,
        polyline: false,
        circlemarker: false
    },
    edit: {
        featureGroup: drawnItems,
        remove: true
    }
});
map.addControl(drawControl);

// Store current rectangle coordinates
let currentRectangle = null;
let heatmapOverlay = null;

// Function to show funky loader
function showLoader() {
    const loader = document.createElement('div');
    loader.id = 'funky-loader';
    loader.innerHTML = `
        <div class="loader-content">
            <div class="loader-spinner">
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
            </div>
            <div class="loader-text">
                <h2>Analyzing Ocean Data...</h2>
                <p class="loading-message">Please wait while we process your request</p>
            </div>
        </div>
    `;
    document.body.appendChild(loader);
}

function hideLoader() {
    const loader = document.getElementById('funky-loader');
    if (loader) {
        document.body.removeChild(loader);
    }
}

// Function to show date input popup
function showDatePopup(callback) {
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';
    document.body.appendChild(overlay);

    const popup = document.createElement('div');
    popup.className = 'date-popup';

    popup.innerHTML = `
        <h3>Enter Date</h3>
        <label>Year: <input type="number" id="yearInput" min="1900" max="2100" required></label>
        <label>Month: <input type="number" id="monthInput" min="1" max="12" required></label>
        <label>Day: <input type="number" id="dayInput" min="1" max="31" required></label>
        <button id="submitDate">Submit</button>
        <button id="cancelDate">Cancel</button>
    `;

    document.body.appendChild(popup);

    document.getElementById('submitDate').addEventListener('click', () => {
        const year = document.getElementById('yearInput').value;
        const month = document.getElementById('monthInput').value;
        const day = document.getElementById('dayInput').value;

        if (year && month && day) {
            const date = new Date(year, month - 1, day);
            if (isNaN(date.getTime()) || year < 1900 || year > 2100 || month < 1 || month > 12 || day < 1 || day > 31) {
                alert('Please enter a valid date (YYYY: 1900-2100, MM: 1-12, DD: 1-31).');
                return;
            }
            document.body.removeChild(popup);
            document.body.removeChild(overlay);
            callback({ year, month, day });
        } else {
            alert('Please fill in all date fields.');
        }
    });

    document.getElementById('cancelDate').addEventListener('click', () => {
        document.body.removeChild(popup);
        document.body.removeChild(overlay);
        callback(null);
    });
}

// Function to toggle layout
function toggleLayout(active) {
    const container = document.querySelector('.container');
    const mapSection = document.querySelector('.map-section');
    const loadAreas = document.querySelector('#load-areas');

    if (active) {
        container.classList.add('active');
        mapSection.classList.add('active');
        loadAreas.classList.add('active');
    } else {
        container.classList.remove('active');
        mapSection.classList.remove('active');
        loadAreas.classList.remove('active');
    }
}

// Calculate distance between two points in kilometers
function calculateDistance(point1, point2) {
    const R = 6371;
    const dLat = (point2.lat - point1.lat) * Math.PI / 180;
    const dLng = (point2.lng - point1.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(point1.lat * Math.PI / 180) * Math.cos(point2.lat * Math.PI / 180) *
        Math.sin(dLng / 2) * Math.sin(dLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// Validate rectangle size
function isValidRectangleSize(bounds) {
    const width = calculateDistance(
        { lat: bounds.getNorthWest().lat, lng: bounds.getNorthWest().lng },
        { lat: bounds.getNorthEast().lat, lng: bounds.getNorthEast().lng }
    );
    const height = calculateDistance(
        { lat: bounds.getNorthWest().lat, lng: bounds.getNorthWest().lng },
        { lat: bounds.getSouthWest().lat, lng: bounds.getSouthWest().lng }
    );
    return width <= 25 && height <= 25;
}

// Function to send data to API and get prediction
async function sendToAPI(bounds, date) {
    const payload = {
        bounds: {
            southWest: {
                lat: bounds.getSouthWest().lat,
                lng: bounds.getSouthWest().lng
            },
            northEast: {
                lat: bounds.getNorthEast().lat,
                lng: bounds.getNorthEast().lng
            }
        },
        date: `${date.year}-${date.month.toString().padStart(2, '0')}-${date.day.toString().padStart(2, '0')}`
    };

    showLoader();

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        hideLoader();
        return data;
    } catch (error) {
        hideLoader();
        alert(`Error: ${error.message}`);
        throw error;
    }
}

// Function to display heatmap on map
function displayHeatmap(base64Image, bounds) {
    // Remove previous heatmap if exists
    if (heatmapOverlay) {
        map.removeLayer(heatmapOverlay);
    }

    // Create image overlay
    const imageBounds = [[bounds.getSouthWest().lat, bounds.getSouthWest().lng],
                         [bounds.getNorthEast().lat, bounds.getNorthEast().lng]];
    
    heatmapOverlay = L.imageOverlay('data:image/png;base64,' + base64Image, imageBounds, {
        opacity: 0.7
    }).addTo(map);
}

// Function to display API response statistics
function displayStatistics(data) {
    const coordDiv = document.getElementById('coordinates');
    const stats = data.statistics;
    const risk = data.risk_distribution;
    
    coordDiv.innerHTML = `
        <h4>Analysis Results</h4>
        <p><span class="coordinate-label">Status:</span> ${data.status}</p>
        <p><span class="coordinate-label">Message:</span> ${data.message}</p>
        <p><span class="coordinate-label">Total Points:</span> ${data.total_points}</p>
        
        <h4>Statistics</h4>
        <p><span class="coordinate-label">Valid Predictions:</span> ${stats.valid_predictions}</p>
        <p><span class="coordinate-label">Min Concentration:</span> ${stats.min.toFixed(4)}</p>
        <p><span class="coordinate-label">Max Concentration:</span> ${stats.max.toFixed(4)}</p>
        <p><span class="coordinate-label">Mean:</span> ${stats.mean.toFixed(4)}</p>
        <p><span class="coordinate-label">Median:</span> ${stats.median.toFixed(4)}</p>
        <p><span class="coordinate-label">Std Dev:</span> ${stats.std.toFixed(4)}</p>
        
        <h4>Risk Distribution</h4>
        <p><span class="coordinate-label">High Risk:</span> ${risk.high_risk.count} (${risk.high_risk.percentage}%)</p>
        <p><span class="coordinate-label">Moderate Risk:</span> ${risk.moderate_risk.count} (${risk.moderate_risk.percentage}%)</p>
        <p><span class="coordinate-label">Low Risk:</span> ${risk.low_risk.count} (${risk.low_risk.percentage}%)</p>
        <p><span class="coordinate-label">Very Low Risk:</span> ${risk.very_low_risk.count} (${risk.very_low_risk.percentage}%)</p>
    `;
}

// Handle rectangle creation
map.on('draw:created', function (e) {
    const layer = e.layer;
    const bounds = layer.getBounds();

    if (!isValidRectangleSize(bounds)) {
        alert('Error: The selected area exceeds 25km x 25km. Please draw a smaller rectangle.');
        return;
    }

    showDatePopup(async (date) => {
        if (!date) {
            return;
        }

        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        currentRectangle = {
            northEast: {
                lat: bounds.getNorthEast().lat,
                lng: bounds.getNorthEast().lng
            },
            southWest: {
                lat: bounds.getSouthWest().lat,
                lng: bounds.getSouthWest().lng
            },
            northWest: {
                lat: bounds.getNorthWest().lat,
                lng: bounds.getNorthWest().lng
            },
            southEast: {
                lat: bounds.getSouthEast().lat,
                lng: bounds.getSouthEast().lng
            },
            center: {
                lat: bounds.getCenter().lat,
                lng: bounds.getCenter().lng
            },
            date: {
                year: parseInt(date.year),
                month: parseInt(date.month),
                day: parseInt(date.day)
            },
            timestamp: new Date().toISOString()
        };

        toggleLayout(true);

        // Send to API
        try {
            const apiResponse = await sendToAPI(bounds, date);
            displayHeatmap(apiResponse.heatmap_base64, bounds);
            displayStatistics(apiResponse);
        } catch (error) {
            console.error('API call failed:', error);
        }
    });
});

// Handle rectangle editing
map.on('draw:edited', function (e) {
    const layers = e.layers;
    layers.eachLayer(function (layer) {
        const bounds = layer.getBounds();

        if (!isValidRectangleSize(bounds)) {
            alert('Error: The edited area exceeds 25km x 25km. Please adjust to a smaller size.');
            drawnItems.clearLayers();
            if (currentRectangle) {
                const validBounds = L.latLngBounds(
                    L.latLng(currentRectangle.southWest.lat, currentRectangle.southWest.lng),
                    L.latLng(currentRectangle.northEast.lat, currentRectangle.northEast.lng)
                );
                const rectangle = L.rectangle(validBounds, {
                    color: '#667eea',
                    weight: 3,
                    fillOpacity: 0.3
                });
                drawnItems.addLayer(rectangle);
            }
            return;
        }

        showDatePopup(async (date) => {
            if (!date) {
                return;
            }

            currentRectangle = {
                northEast: {
                    lat: bounds.getNorthEast().lat,
                    lng: bounds.getNorthEast().lng
                },
                southWest: {
                    lat: bounds.getSouthWest().lat,
                    lng: bounds.getSouthWest().lng
                },
                northWest: {
                    lat: bounds.getNorthWest().lat,
                    lng: bounds.getNorthWest().lng
                },
                southEast: {
                    lat: bounds.getSouthEast().lat,
                    lng: bounds.getSouthEast().lng
                },
                center: {
                    lat: bounds.getCenter().lat,
                    lng: bounds.getCenter().lng
                },
                date: {
                    year: parseInt(date.year),
                    month: parseInt(date.month),
                    day: parseInt(date.day)
                },
                timestamp: new Date().toISOString()
            };
            
            toggleLayout(true);

            try {
                const apiResponse = await sendToAPI(bounds, date);
                displayHeatmap(apiResponse.heatmap_base64, bounds);
                displayStatistics(apiResponse);
            } catch (error) {
                console.error('API call failed:', error);
            }
        });
    });
});

// Handle rectangle deletion
map.on('draw:deleted', function () {
    currentRectangle = null;
    if (heatmapOverlay) {
        map.removeLayer(heatmapOverlay);
        heatmapOverlay = null;
    }
    document.getElementById('coordinates').innerHTML = '<p>No area selected yet. Use the rectangle tool to draw on the map.</p>';
    toggleLayout(false);
});

// Save button now sends to API
document.getElementById('saveBtn').addEventListener('click', async function () {
    if (!currentRectangle) {
        alert('Please draw a rectangle on the map first!');
        return;
    }

    const bounds = L.latLngBounds(
        L.latLng(currentRectangle.southWest.lat, currentRectangle.southWest.lng),
        L.latLng(currentRectangle.northEast.lat, currentRectangle.northEast.lng)
    );

    try {
        const apiResponse = await sendToAPI(bounds, currentRectangle.date);
        displayHeatmap(apiResponse.heatmap_base64, bounds);
        displayStatistics(apiResponse);
        alert('Area processed successfully!');
    } catch (error) {
        console.error('API call failed:', error);
    }
});

// Clear selection
document.getElementById('clearBtn').addEventListener('click', function () {
    drawnItems.clearLayers();
    if (heatmapOverlay) {
        map.removeLayer(heatmapOverlay);
        heatmapOverlay = null;
    }
    currentRectangle = null;
    document.getElementById('coordinates').innerHTML = '<p>No area selected yet. Use the rectangle tool to draw on the map.</p>';
    toggleLayout(false);
});

// Load and Delete buttons removed functionality (no localStorage)
document.getElementById('loadBtn').addEventListener('click', function () {
    alert('Load feature removed - data is now sent directly to API');
});

document.getElementById('deleteBtn').addEventListener('click', function () {
    alert('Delete feature removed - no local storage used');
});

document.getElementById('savedList').innerHTML = '<p>Local storage disabled - all data sent to API</p>';