# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Mixin + decorator for topology passes that need the platform's engine list.

A lowering pass that wants to assign nodes to specific engines (e.g. a
spatial-split pass that places sub-nodes on individual AIE cores) needs to
know which engines exist. The pass declares its dependency by wearing
:func:`engineaware`, and :class:`EngineColoringDeployer` injects the engine
list via :meth:`setEngines` before running the optimizer.
"""

from __future__ import annotations

from typing import List

from Deeploy.DeeployTypes import DeploymentEngine


class EngineAwarePassMixIn:
    """Marks a pass as needing the platform's engine list.

    The orchestrator (typically :class:`EngineColoringDeployer`) calls
    :meth:`setEngines` before invoking ``apply``. The pass then reads
    ``self.engines`` from inside ``apply``.
    """

    def setEngines(self, engines: List[DeploymentEngine]) -> None:
        self.engines = list(engines)


def engineaware(cls):
    """Mix :class:`EngineAwarePassMixIn` into ``cls``.

    Parallels the existing :func:`contextaware` / :func:`contextagnostic`
    decorators in :mod:`Deeploy.CommonExtensions.OptimizationPasses.PassClasses`
    but is orthogonal to them: ``apply``'s signature is unchanged.
    """
    return type(cls.__name__, (cls, EngineAwarePassMixIn), {})
