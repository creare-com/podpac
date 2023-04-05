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

