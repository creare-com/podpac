
from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import podpac
from podpac.core.pipeline.pipeline import Pipeline, make_pipeline_definition

def test_make_pipeline_definition():
    a = podpac.core.algorithm.algorithm.Arange()
    b = podpac.core.algorithm.algorithm.CoordData()
    c = podpac.core.compositor.OrderedCompositor(sources=np.array([a, b]))
    d = podpac.core.algorithm.algorithm.Arithmetic(A=a, B=b, C=c, eqn="A + B + C")
    
    definition = make_pipeline_definition(d)

    # make sure it is a valid pipeline
    pipeline = Pipeline(definition)

    assert isinstance(pipeline.nodes[a.base_ref], podpac.core.algorithm.algorithm.Arange)
    assert isinstance(pipeline.nodes[b.base_ref], podpac.core.algorithm.algorithm.CoordData)
    assert isinstance(pipeline.nodes[c.base_ref], podpac.core.compositor.OrderedCompositor)
    assert isinstance(pipeline.nodes[d.base_ref], podpac.core.algorithm.algorithm.Arithmetic)
    assert pipeline.output == None

def test_make_pipeline_definition_duplicate_base_ref():
    a = podpac.core.algorithm.algorithm.Arange()
    b = podpac.core.algorithm.algorithm.Arange()
    c = podpac.core.algorithm.algorithm.Arange()
    d = podpac.core.compositor.OrderedCompositor(sources=np.array([a, b, c]))
    
    definition = make_pipeline_definition(d)

    # make sure it is a valid pipeline
    pipeline = Pipeline(definition)

    assert len(pipeline.nodes) == 4