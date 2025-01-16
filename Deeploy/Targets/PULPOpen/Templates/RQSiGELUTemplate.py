from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _PULPRQSiGELUTemplate(NodeTemplate):
    """
    Parallelized version of RQSiGELU for PULPOpen.
    Uses PULPiGELU_s8_s8(...) from iGELU.c to do the actual iGELU operation,
    splitting the data across cores.
    """

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(
        self,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        operatorRepresentation['mul_scalar']   = operatorRepresentation['mul']
        operatorRepresentation['add_scalar']   = operatorRepresentation['add']
        operatorRepresentation['shift_scalar'] = operatorRepresentation['shift']

        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPRQSiGELUTemplate("""
/* Requantized iGELU parallelized for PULP (Name: ${nodeName}, Op: ${nodeOp}) */

PULPiGELU_s8_s8(
    ${data_in},             /* entire base pointer for input  */
    ${data_out},            /* entire base pointer for output */
    ${size},                /* full number of elements        */
    ${b},                   /* 'b' param (int8)               */
    ${one},                 /* 'one' param (int16)            */
    ${input_offset},        /* input offset                   */
    ${output_offset},       /* output offset                  */
    &${mul_scalar}[0],      /* pointer to 'mul' scalar        */
    &${add_scalar}[0],      /* pointer to 'add' scalar        */
    &${shift_scalar}[0]     /* pointer to 'shift' scalar      */
);

pi_cl_team_barrier();
""")