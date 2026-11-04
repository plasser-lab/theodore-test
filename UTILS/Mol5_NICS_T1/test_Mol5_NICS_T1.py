#!/usr/bin/env python3
"""
NICS/VIST test for an open-shell triplet (Gaussian 09, UPBE1PBE/def2-SVP NMR,
T1 state, 6 Bq atoms on the axes of the two five-membered rings).
"""

import sys
import numpy
import pytest
from theodore import lib_pytest, lib_NICS
from theodore.lib_pytest import ActionFactory, mock_stdout

RING_A = [1, 6, 2, 8, 9]
RING_B = [2, 12, 3, 10, 8]

def test_Mol5_NICS_T1():
    pjob = lib_pytest.pytest_job(__file__)

    # NICS.txt, NICS_ring.txt (ring A) and VIST.vmd
    with mock_stdout() as out:
        sys.argv = ['theodore', 'plot_vist', '-r', ' '.join(str(i) for i in RING_A), 'gaussian.log']
        ActionFactory.from_commandline()

    nv = lib_NICS.NICS_parser_gaussian()
    nv.read('gaussian.log', lvprt=0)
    assert len(nv.NICS_data) == 6
    assert len(nv.atoms) == 17
    # isotropic values as printed in the log file
    assert [p.NICS_iso for p in nv.NICS_data] == [11.4948, 12.2383, 12.2327, 3.6389, 7.0877, 7.0834]
    for p in nv.NICS_data:
        assert numpy.trace(p.NICS_tensor) / 3. == pytest.approx(p.NICS_iso, abs=1.e-4)

    res_A = nv.ring_analysis(RING_A, outfile=None, lvprt=0)
    res_B = nv.ring_analysis(RING_B, outfile='NICS_ring_B.txt', lvprt=0)
    assert [r['point'] for r in res_A] == [0, 1, 2]
    assert [r['point'] for r in res_B] == [3, 4, 5]
    for res in res_A, res_B:
        assert [r['label'] for r in res] == ['NICS(0)', 'NICS(1)', 'NICS(-1)']
        # planar molecule: at NICS(0) the ring normal is an eigenvector of the tensor
        assert res[0]['eig_angle'] < 0.1
        assert res[0]['NICS_eig'] == pytest.approx(res[0]['NICS_zz'], abs=1.e-3)
        # above and below the ring NICS_eig and NICS_zz differ slightly
        for r in res[1:]:
            assert r['eig_angle'] < 10.
            assert r['NICS_eig'] == pytest.approx(r['NICS_zz'], abs=0.5)
    # aromatic / antiaromatic character of the two rings in the T1 state
    assert res_A[1]['NICS_zz'] == pytest.approx(-28.3658, abs=1.e-4)
    assert res_B[0]['NICS_zz'] == pytest.approx(10.8117, abs=1.e-4)

    pjob.check()
