"""
SoilGrids

See: https://maps.isric.org/
"""


from podpac.data import WCS


class SoilGridsBase(WCS):
    """Base SoilGrids WCS datasource."""

    format = "geotiff_byte"
    max_size = 16384
    _repr_keys = ["layer"]


class SoilGridsWRB(SoilGridsBase):
    """SoilGrids: WRB classes and probabilities (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/wrb.map"


class SoilGridsBDOD(SoilGridsBase):
    """SoilGrids: Bulk density (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/bdod.map"


class SoilGridsCEC(SoilGridsBase):
    """SoilGrids: Cation exchange capacity and ph 7 (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/cec.map"


class SoilGridsCFVO(SoilGridsBase):
    """SoilGrids: Coarse fragments volumetric (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/cfvo.map"


class SoilGridsClay(SoilGridsBase):
    """SoilGrids: Clay content (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/clay.map"


class SoilGridsNitrogen(SoilGridsBase):
    """SoilGrids: Nitrogen (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/nitrogen.map"


class SoilGridsPHH2O(SoilGridsBase):
    """SoilGrids: Soil pH in H2O (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/phh2o.map"


class SoilGridsSand(SoilGridsBase):
    """SoilGrids: Sand content (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/sand.map"


class SoilGridsSilt(SoilGridsBase):
    """SoilGrids: Silt content (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/silt.map"


class SoilGridsSOC(SoilGridsBase):
    """SoilGrids: Soil organic carbon content (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/soc.map"


class SoilGridsOCS(SoilGridsBase):
    """SoilGrids: Soil organic carbon stock (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/ocs.map"


class SoilGridsOCD(SoilGridsBase):
    """SoilGrids: Organic carbon densities (WCS)"""

    source = "https://maps.isric.org/mapserv?map=/map/ocd.map"


if __name__ == "__main__":
    import podpac

    c = podpac.Coordinates(
        [podpac.clinspace(-132.9023, -53.6051, 346, name="lon"), podpac.clinspace(23.6293, 53.7588, 131, name="lat")]
    )

    print("layers")
    print(SoilGridsSand.get_layers())

    node = SoilGridsSand(layer="sand_0-5cm_mean")
    print("node")
    print(node)

    output = node.eval(c)
    print("eval")
    print(output)

    node_chunked = SoilGridsSand(layer="sand_0-5cm_mean", max_size=10000)
    output_chunked = node_chunked.eval(c)

    from matplotlib import pyplot

    pyplot.figure()
    pyplot.subplot(211)
    output.plot()
    pyplot.subplot(212)
    output_chunked.plot()
    pyplot.show(block=False)
