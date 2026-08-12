#!/usr/bin/env python3

import os
from theodore import lib_pytest

def test_standard():
    pjob = lib_pytest.pytest_job(__file__)

    stdin = open('../IN_FILES/plot_omfrag.in', 'r').read()
    pjob.run_util('plot_omfrag', stdin)

    stdin = open('../IN_FILES/plot_om_bars.in', 'r').read()
    pjob.run_util('plot_om_bars', stdin)

    stdin = open('../IN_FILES/plot_frag_decomp.in', 'r').read()
    pjob.run_util('plot_frag_decomp', stdin)

    stdin = open('../IN_FILES/spec_eps.in', 'r').read()
    pjob.run_util('spectrum tden_summ.txt', stdin)
    os.rename('spectrum.dat', 'spectrum-eps.dat')

    stdin = open('../IN_FILES/spectrum.in', 'r').read()
    pjob.run_util('spectrum tden_summ.txt tden_summ2.txt', stdin)

    stdin = open('../IN_FILES/convert_table.in', 'r').read()
    pjob.run_util('convert_table', stdin)

    pjob.check()
