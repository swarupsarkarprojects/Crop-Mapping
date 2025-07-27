# Crop-Mapping using Senitnel and and Random Forest Classifer method in Google Colab Python:
This project demonstrates crop mapping using Sentinel‑2 surface reflectance imagery with Google Earth Engine and Python on Google Colab.
It uses vegetation indices (NDVI, NDWI, EVI) and a Random Forest classifier to distinguish crop vs non‑crop areas, then exports the classified raster.

# Features:
✅ Uses Google Earth Engine Python API and geemap
✅ Fetches recent Sentinel‑2 imagery within a specified AOI
✅ Computes vegetation indices (NDVI, NDWI, EVI)
✅ Creates training points for crop and non‑crop areas
✅ Trains and evaluates a Random Forest model in Python
✅ Classifies the imagery and exports the result to Google Drive
✅ Interactive visualization using geemap

# Requirements:
 1. Google account with access to Google Earth Engine
 2. Run in Google Colab or local Jupyter Notebook
 3. Python libraries:
pip install geemap --quiet
pip install --upgrade geemap
import ee, geemap

# Authenticate with GEE API:
ee.Authenticate()  
ee.Initialize(project='ee-sarkarswarup')

# Define Area of Intrest:
aoi = ee.Geometry.Polygon([
    [[88.3, 22.7], [88.6, 22.7], [88.6, 22.9], [88.3, 22.9], [88.3, 22.7]]
])

# Time range (recent season)
start = '2024-11-01'
end   = '2025-02-28'

# Sentinel-2 Surface Reflectance
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2','B3','B4','B8']))

# Median composite and AOI clip (or use monthly composites for time series)
img = s2.median().clip(aoi)
img_normalized = img.expression(
    '((image)/10000)',{
        'image': img
    }
)
print(img_normalized.getInfo())

# Calculate Vegetation Indices and add with image band: 
def addIndices(image):
    ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3','B8']).rename('NDWI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
        }).rename('EVI')
    return image.addBands(ndvi).addBands(ndwi).addBands(evi)

bands = ['B2','B3','B4','B8','NDVI','NDWI','EVI']
img_indices = addIndices(img_normalized)

print(img_indices.getInfo())

# Indentify the Crop and Non-Crop pixel:
ndvi = img_indices.select('NDVI')
crop_mask = (ndvi.gt(0.2)) and (ndvi.lt(0.4))
noncrop_mask = ndvi.lt(0.1)

# Create Random points for Training Datasets
crop_points = crop_mask.stratifiedSample(
    numPoints=100, classBand='NDVI', region=aoi, scale=10, seed=42, geometries=True)
noncrop_points = noncrop_mask.stratifiedSample(
    numPoints=100, classBand='NDVI', region=aoi, scale=10, seed=84, geometries=True)

# Assign labels
crop_points = crop_points.map(lambda f: f.set('class', 1))
noncrop_points = noncrop_points.map(lambda f: f.set('class', 0))

# Merge 
training_points = crop_points.merge(noncrop_points)

training = img_indices.sampleRegions(
    collection=training_points,
    properties=['class'],
    scale=10
)
print(training.getInfo())
features = training.getInfo()['features']

# Converting Feature to Pandas Feature for model run:
import pandas as pd
training_df = pd.json_normalize(features)
print(training_df.columns)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Preparing poit datasets for calculate the model accuracy:
bands2 = [
    'properties.B2','properties.B3','properties.B4',
    'properties.B8','properties.NDVI','properties.NDWI','properties.EVI'
]
X = training_df[bands2].values
y = training_df['properties.class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

# Run the Random Forest model:
rf = ee.Classifier.smileRandomForest(100).train(
    features=training,
    classProperty='class',
    inputProperties=bands
)

classified = img_indices.classify(rf)

# Exporting the final result on Drive:
task = ee.batch.Export.image.toDrive(
    image=classified,
    description='crop_mapping_result',
    folder='earthengine',
    region=aoi,
    scale=10,
    maxPixels=1e13
)
task.start()

# Visualization of Result:
Map = geemap.Map(center=[22.8,88.45], zoom=11)
Map.addLayer(img_indices, {'bands':['B8','B4','B3'], 'min':0, 'max':0.3}, 'RGB')
Map.addLayer(classified, { 'min':0, 'max':1, 'palette':['red','green'] }, 'Crop Mapping')
Map
