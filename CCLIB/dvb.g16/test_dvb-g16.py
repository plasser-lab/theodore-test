#!/usr/bin/env python3

import os
import pytest
from theodore import lib_pytest, error_handler
from theodore.lib_pytest import ActionFactory, commandline, mock_stdout

def test_standard():
    lib_pytest.pytest_job(__file__).run_standard()

def test_inconsistent_mo_file():
    """The MO file must belong to the same SCF solution as the TDDFT job."""
    pjob = lib_pytest.pytest_job(__file__)
    with open('dvb_sp.out') as f:
        lines = f.read().replace('E(RB3LYP) =  -382.308266602', 'E(RB3LYP) =  -382.300000000')
    with open('dvb_sp_other.out', 'w') as f:
        f.write(lines)
    with open('../IN_FILES/tden.in.g16') as f:
        inp = f.read().replace("dvb_sp.out", "dvb_sp_other.out")
    with open('tden.in.bad', 'w') as f:
        f.write(inp)
    with commandline('theodore analyze_tden -f tden.in.bad'), mock_stdout():
        with pytest.raises(error_handler.MsgError):
            ActionFactory.from_commandline()
