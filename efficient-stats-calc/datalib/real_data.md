# Real Data

Need: 
- Soil Density (geowatch)
LAYERS=algorithm.standard_mobility.soil_density.SoilDensityComposite
- Soil Type (gw)
- Soil Texture (gw)?
LAYERS=datalib.soil.texture.USCSTextureComposite
- Soil Temperature (gw)
LAYERS=datalib.weather.erdc_lis72.SoilTemperature0to10
- Soil Moisture (soilmap)


## Example

```
c=podpac.Coordinates([podpac.clinspace([43.700, 43.704, 128, 'lat'], podpac.clinspace([-72.305, -72.301, 128, 'lon'], '2023-01-01', ['lat', 'lon', 'time'])

sm = soilmap.datalib.geowatch.SoilMoisture()
sm.eval(c)

podpac.settings["username-PW@geowatch_server"] = {"username": "creare", "password": "<iwillsayverbally"}
podpac.settings.save()
```

## TODO
1. Wrap Soil Strength
2. Fix Caching 
3. Test on small region



1. Pull multiple year soilmoisture coarse resolution for 3x3 pixels
only do work in the middle pixel
SoilMoisture.solmst0_to_10

c=podpac.Coordinates([podpac.clinspace(43.09, 42.91, 3, 'lat'), podpac.clinspace(-73.155, -72.91, 3, 'lon'), '2023-01-01'], ['lat', 'lon', 'time'])
- correct resolution?
2. Cache it

3. Feed 


## TODO PT 2
1. Get weatherscale resolution data for every day of a month on a single pixel
2. Get construct a CDF as expected output
3. Get fine resolution data for that pixel at the fine resolution for every day of a month
4. Construct a CDF as input to model

