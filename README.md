# DepthWizard - SIH 2026
Team project for Smart India Hackathon 2026, Problem Statement 26175.
We are creating accurate 3D maps of land and buildings usually requires expensive equipment like LiDAR or multiple satellite images.
**DepthWizard** simplifies this by creating a 3D representation from just one aerial or satellite image. Our system uses Artificial Intelligence to estimate the relative height of objects such as buildings, trees, roads, and hills.
For **georeferenced GeoTIFF images**, which contain location information, a **scale-calibration module** converts these relative depth estimates into real-world heights in metres. It uses low-resolution elevation data such as SRTM (a global elevation dataset), a few known Ground Control Points, or semantic scene priors (information about what objects are likely to be and their typical heights) to produce an *Absolute Digital Surface Model (DSM)*. 
For normal JPG or PNG images, the system generates relative heights for **3D viewing**. Finally, the height map is combined with the original image to create an **interactive 3D environment** for inspecting heights and slopes.
