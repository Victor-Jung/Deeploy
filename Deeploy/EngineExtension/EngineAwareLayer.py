# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""ONNXLayer subclass that transfers engine coloring from graph nodes into operatorRepresentation."""

from __future__ import annotations

from typing import Dict, Tuple, Type

from Deeploy.DeeployTypes import NetworkContext, ONNXLayer


class EngineAwareONNXLayer(ONNXLayer):
    """ONNXLayer subclass that lifts ``node.attrs['engine']`` into
    ``operatorRepresentation['engine']`` after the base parse succeeds.
    """

    def parse(self, ctxt: NetworkContext, default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        newCtxt, ok = super().parse(ctxt, default_channels_first)
        if ok:
            self.mapper.parser.operatorRepresentation['engine'] = self.node.attrs.get('engine')
        return newCtxt, ok


def engineAware(layerCls: Type[ONNXLayer]) -> Type[ONNXLayer]:
    """Return an engine-aware subclass of ``layerCls``.

    If ``layerCls`` already derives from :class:`EngineAwareONNXLayer`,
    it's returned unchanged. Otherwise a new class with MRO
    ``(EngineAwareONNXLayer, layerCls, …)`` is constructed; the lift runs
    after ``layerCls.parse`` completes via standard ``super()`` chaining.
    """

    if issubclass(layerCls, EngineAwareONNXLayer):
        return layerCls
    name = f"EngineAware{layerCls.__name__}"
    return type(name, (EngineAwareONNXLayer, layerCls), {})


def makeMappingEngineAware(mapping: Dict[str, ONNXLayer]) -> Dict[str, ONNXLayer]:
    """Return a new platform-mapping dict where every layer is engine-aware."""
    
    return {op: engineAware(type(layer))(layer.maps) for op, layer in mapping.items()}
