
"""
Pipeline utils
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import re

def make_pipeline_definition(main_node):
    """
    Make a pipeline definition, including the flattened node definitions and a
    default file output for the input node.

    Parameters
    ----------
    main_node : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    nodes = []
    refs = []
    definitions = []

    def add_node(node):
        """Summary

        Parameters
        ----------
        node : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        if node in nodes:
            return refs[nodes.index(node)]

        # get definition
        d = node.definition

        # replace nodes with references, adding nodes depth first
        if 'inputs' in d:
            for key, input_node in d['inputs'].items():
                if input_node is not None:
                    d['inputs'][key] = add_node(input_node)
        if 'sources' in d:
            for i, source_node in enumerate(d['sources']):
                d['sources'][i] = add_node(source_node)

        # unique ref
        ref = node.base_ref
        if ref in refs:
            if re.search('_[1-9][0-9]*$', ref):
                ref, i = ref.rsplit('_', 1)
                i = int(i)
            else:
                i = 0
            ref = '%s_%d' % (ref, i+1)

        nodes.append(node)
        refs.append(ref)
        definitions.append(d)

        return ref

    add_node(main_node)

    output = OrderedDict()
    #output['mode'] = 'file'
    #output['format'] = 'pickle'
    #output['outdir'] = os.path.join(os.getcwd(), 'out')
    #output['nodes'] = [refs[-1]]

    output['mode'] = 'image'
    output['format'] = 'png'
    output['vmin'] = -1.2
    output['vmax'] = 1.2
    output['nodes'] = [refs[-1]]


    d = OrderedDict()
    d['nodes'] = OrderedDict(zip(refs, definitions))
    d['outputs'] = [output]
    return d