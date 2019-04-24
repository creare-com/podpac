t_coords = deepcopy(self)

        if crs is None:
            return self.copy()

        from_crs = pyproj.CRS(self.crs)
        to_crs = pyproj.CRS(crs)

        # convert alt units into proj4 syntax
        if alt_units is not None:
            proj4crs = to_crs.to_proj4()

            # remove old vunits string if present
            if '+vunits' in proj4crs:
                proj4crs = re.sub(r'\+vunits=[a-z\-]+\s', '', proj4crs)

            to_crs = pyproj.CRS('{} +vunits={}'.format(proj4crs, alt_units))

        # create proj4 transformer
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs)

        # create coords to be transformed
        t_coords = deepcopy(self)

        # update crs on the individual coords - this must be done before assigning new values
        # note using `srs` here so it captures the user input (i.e. EPSG:4193)
        # if alt_units included, this will be a whole proj4 string because of conversion above
        for dim in t_coords.udims:
            t_coords[dim].crs = to_crs.srs

        # Try to convert lat, lon to DependentCoordinates
        cs = [c for c in self.values()]
        if 'lat' in self.dims and 'lon' in self.dims:
            lat = np.array([self['lat'].coordinates for i in range(self['lon'].size)])
            lon = np.array([self['lon'].coordinates for i in range(self['lat'].size)]).T
            ilat = self.dims.index('lat')
            ilon = self.dims.index('lon')
            if ilat == ilon-1:
                cs.pop(ilon)
                cs.pop(ilat)
                cs.insert(ilat, DependentCoordinates([lat, lon], dims=['lat', 'lon'], etc))
            elif ilon == ilat-1:
                cs.pop(ilat)
                cs.pop(ilon)
                cs.insert(ilon, DependentCoordinates([lon.T, lat.T], dims=['lon', 'lat'], etc))
            else:
                raise ValueError()

        # transform
        ts = [c._transform(transformer) for c in cs]
        ts = []
        for c in cs:
            if isinstance(c, Coordinates1d):
                if c.name in ['lat', 'lon']:
                    raise ValueError()
                if c.name == 'alt':
                    dummy = np.zeros(len(self.coords['alt'].values))  # must be same length as alt
                    (lat, lon, alt) = transformer.transform(dummy, dummy, self.coords['alt'].values)
                    properties = c.properties
                    properties['crs'] = to_crs.srs
                    t = ArrayCoordinates1d(alt, **properties) # TODO return UniformCoordinates when possible
                elif if c.name == 'time':
                    t = c # don't transform
            elif isinstance(c, StackedCoordinates):
                (lat, lon) = transformer.transform(c['lat'].coordinates, c['lon'].coordinates)
                t = c.copy()
                t['lat'].set_trait('coordinates', lat)
                t['lon'].set_trait('coordinates', lon)
                t['lat'].set_trait('crs', to_crs.srs)
                t['lon'].set_trait('crs', to_crs.srs)
            elif isinstance(c, DependentCoordinates):
                (lat, lon) = transformer.transform(c['lat'].coordinates, c['lon'].coordinates)
                t = c.copy()
                t.coordinates[c.dims.index('lat')] = lat
                t.coordinates[c.dims.index('lon')] = lon
                t.set_trait('crs', to_crs.srs)
                t = DependentCoordinates()
            ts.append(t)
        return Coordinates(ts)


        # if lat or lon is present, coordinates MUST have both, even if stacked:
        if 'lat_lon' in self.dims or 'lon_lat' in self.dims:
            (lat, lon) = transformer.transform(self['lat'].coordinates, self['lon'].coordinates)
            t_coords['lat'] = ArrayCoordinates1d(lat, crs=t_coords.crs)
            t_coords['lon'] = ArrayCoordinates1d(lon, crs=t_coords.crs)

        elif 'lat' in self.dims and 'lon' in self.dims:
            ilat = self.dims.index('lat')
            ilon = self.dims.index('lon')

            if ilat < ilon:
                lat = np.array([self['lat'].coordinates for i in range(self['lon'].size)])
                lon = np.array([self['lon'].coordinates for i in range(self['lat'].size)]).T
                (lat, lon) = transformer.transform(lat, lon)
                t_coords._coords['lat,lon'] = DependentCoordinates([lat, lon], dims=['lat', 'lon'])
            else:
                lat = np.array([self['lon'].coordinates for i in range(self['lon'].size)])
                lon = np.array([self['lat'].coordinates for i in range(self['lat'].size)]).T
                (lat, lon) = transformer.transform(lat, lon)
                t_coords._coords['lon,lat'] = DependentCoordinates([lon, lat], dims=['lon', 'lat'])
            
            del t_coords._coords['lat']
            del t_coords._coords['lon']
            
        elif 'lat' in self.udims or 'lon' in self.udims:
            raise ValueError('Coordinates must have both lat and lon dimensions to transform coordinate reference systems')

        # by keeping these seperate, we can handle altitude dimensions that are a different length from lat/lon
        if 'alt' in self.dims:
  
            dummy = np.zeros(len(self.coords['alt'].values))  # must be same length as alt
            (lat, lon, alt) = transformer.transform(dummy, dummy, self.coords['alt'].values)
            t_coords['alt'] = ArrayCoordinates1d(alt, crs=t_coords.crs)

        return t_coords