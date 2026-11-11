#!/usr/bin/env python3
"""
NICS/VIST test for a Gaussian 16 NMR log file (PBE0/def2-TZVP,
[2.2.2.2]paracyclophanetetraene, closed shell, 3 Bq atoms on the axis
of the macrocycle).
"""

import sys
import numpy
import pytest
from theodore import lib_pytest, lib_NICS
from theodore.lib_pytest import ActionFactory, mock_stdout

# all 32 C atoms of the macrocycle, sorted by their angle around the ring axis
RING = [19, 16, 5, 1, 2, 10, 4, 6, 9, 8, 15, 22, 24, 33, 36, 34,
        38, 55, 53, 51, 52, 48, 45, 47, 43, 41, 17, 28, 25, 30, 26, 20]
# one of the benzene rings, without ghost atoms on its axis
BENZENE = [1, 2, 10, 4, 6, 9]

def test_PCT_NICS_G16():
    pjob = lib_pytest.pytest_job(__file__)

    # NICS.txt, NICS_ring.txt, VIST.vmd and coor0.xyz
    with mock_stdout() as out:
        sys.argv = ['theodore', 'plot_vist', '-c', '-r', ' '.join(str(i) for i in RING), 'PCT_gaussian.log']
        ActionFactory.from_commandline()

    nv = lib_NICS.NICS_parser_gaussian()
    nv.read('PCT_gaussian.log', lvprt=0)
    assert len(nv.NICS_data) == 3
    assert len(nv.atoms) == 56
    # isotropic values as printed in the log file
    assert [p.NICS_iso for p in nv.NICS_data] == [-0.4860, -0.5831, -0.5838]
    for p in nv.NICS_data:
        assert numpy.trace(p.NICS_tensor) / 3. == pytest.approx(p.NICS_iso, abs=1.e-4)

    res = nv.ring_analysis(RING, outfile=None, lvprt=0)
    assert [r['point'] for r in res] == [0, 1, 2]
    assert [r['label'] for r in res] == ['NICS(0)', 'NICS(-1)', 'NICS(1)']
    # the ghost atoms are about 1 deg off the axis of the macrocycle
    for r in res:
        assert r['offset'] < 0.025
        assert r['eig_angle'] < 0.5
    assert res[0]['NICS_zz'] == pytest.approx(10.8555, abs=1.e-4)
    assert res[0]['NICS_eig'] == pytest.approx(res[0]['NICS_zz'], abs=1.e-3)

    # no ghost atoms on the axis of the benzene ring
    assert nv.ring_analysis(BENZENE, outfile=None, lvprt=0) == []

    pjob.check()
