#!/usr/bin/env python3
"""
Tests for the Gaussian NICS reader and the NICS ring helpers in lib_NICS.

All input is derived at run time from UTILS/BCyc_VIST/QC_FILES/neutral.log
(Gaussian 09, PBE1PBE/def2-SVP NMR, 8 Bq atoms, input orientation only):
  - a "Standard orientation" variant: the coordinates of a rotated and shifted
    frame are added as standard orientation and all shielding tensors are
    rotated accordingly, as Gaussian does without NoSymm
  - Bq atoms given with freeze codes and without orientation tables
  - Bq atoms given as Z-matrix lines
  - a two-step file (as for Opt+NMR or --Link1--)
"""

import os
import sys
import numpy
import pytest
from theodore import lib_NICS
from theodore.lib_pytest import ActionFactory, mock_stdout

QCDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'BCyc_VIST', 'QC_FILES')
LOG = os.path.join(QCDIR, 'neutral.log')
# Six-membered ring around the Bq atoms P3 (centroid) and P4/P5 (+-1 A)
RING = [4, 5, 7, 9, 8, 6]


def parse(path):
    nv = lib_NICS.NICS_parser_gaussian()
    nv.read(path, lvprt=0)
    return nv

def write(tmp_path, name, lines):
    path = str(tmp_path / name)
    with open(path, 'w') as f:
        f.writelines(lines)
    return path

def rotation(axis, angle):
    axis = numpy.array(axis, float) / numpy.linalg.norm(axis)
    K = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return numpy.eye(3) + numpy.sin(angle) * K + (1 - numpy.cos(angle)) * numpy.dot(K, K)

def orientation_block(lines):
    """Return (start, end) line indices of the table rows of the Input orientation."""
    ihead = [i for i, l in enumerate(lines) if 'Input orientation:' in l][0]
    start = ihead + 5
    end = start
    while not '----' in lines[end]:
        end += 1
    return ihead, start, end

def zmat_bq_lines(lines):
    izmat = [i for i, l in enumerate(lines) if 'Symbolic Z-matrix:' in l][0]
    inds = []
    i = izmat + 1
    while lines[i].strip():
        if lines[i].split()[0] == 'Bq':
            inds.append(i)
        i += 1
    return inds

@pytest.fixture(scope='module')
def ref():
    return parse(LOG)


def test_reference_file(ref):
    assert len(ref.NICS_data) == 8
    assert len(ref.atoms) == 56
    # full precision of the orientation table (the input echo has 5 decimals)
    numpy.testing.assert_allclose(ref.NICS_data[3].orig, [1.820883, 3.270885, -0.081708])
    assert ref.NICS_data[3].ret_NICS_iso() == pytest.approx(-6.4439)

def test_ring_analysis(ref):
    res = ref.ring_analysis(RING, outfile=None, lvprt=0)
    labels = {r['point']: r['label'] for r in res}
    assert labels == {3: 'NICS(0)', 4: 'NICS(-1)', 5: 'NICS(1)'}
    vals = {r['point']: r for r in res}
    assert vals[3]['NICS_iso'] == pytest.approx(-6.4439)
    assert vals[4]['NICS_iso'] == pytest.approx(-7.5935)
    assert vals[5]['NICS_iso'] == pytest.approx(-9.7510)
    # NICS_zz = -n.sigma.n, checked against the tensor printed in the log file
    n = lib_NICS.ring_frame([ref.atoms[i-1]['coor'] for i in RING])[1]
    sigma = ref.NICS_data[3].NICS_tensor
    assert vals[3]['NICS_zz'] == pytest.approx(-numpy.dot(n, numpy.dot(sigma, n)))
    # NICS_zz is also given by the symmetric part only
    assert vals[3]['NICS_zz'] == pytest.approx(-numpy.dot(n, numpy.dot(0.5*(sigma + sigma.T), n)))
    # NICS_eig: eigenvalue whose eigenvector is closest to the ring normal
    for r in res:
        point = ref.NICS_data[r['point']]
        mu = numpy.argmax([abs(numpy.dot(n, point.coor[:, i])) / numpy.linalg.norm(point.coor[:, i]) for i in range(3)])
        assert r['NICS_eig'] == pytest.approx(-point.evals[mu])
    # Above and below the ring the out-of-plane eigenvector is close to the
    # ring normal, at the centroid of this cyclophane it is not
    assert vals[4]['eig_angle'] < 15. and vals[5]['eig_angle'] < 15.
    assert vals[3]['eig_angle'] > 30.
    assert vals[5]['NICS_eig'] == pytest.approx(-24.1712, abs=1.e-4)

def test_ring_from_coordinates(ref):
    coords = [ref.atoms[i-1]['coor'] for i in RING]
    res_ind = ref.ring_analysis(RING, outfile=None, lvprt=0)
    res_coor = ref.ring_analysis(coords, outfile=None, lvprt=0)
    for a, b in zip(res_ind, res_coor):
        assert a['NICS_zz'] == pytest.approx(b['NICS_zz'])

def test_probe_points_match_input(ref):
    """The Bq atoms of the input were placed at the ring centroid and at +-1 A."""
    coords = [ref.atoms[i-1]['coor'] for i in RING]
    points, heights = lib_NICS.ring_probe_points(coords, (0., 1.), both_sides=True)
    assert heights == [0., 1., -1.]
    numpy.testing.assert_allclose(points[0], ref.NICS_data[3].orig, atol=1.e-5)
    # P5 is at +1 A, P4 at -1 A (within the 0.008 A used to set up the input)
    numpy.testing.assert_allclose(points[1], ref.NICS_data[5].orig, atol=1.e-2)
    numpy.testing.assert_allclose(points[2], ref.NICS_data[4].orig, atol=1.e-2)

def test_scan_points():
    ring = [[numpy.cos(a), numpy.sin(a), 0.] for a in numpy.arange(6) * numpy.pi / 3]
    points, heights = lib_NICS.ring_scan_points(ring, 0., 2., 0.5)
    assert heights == [0., 0.5, 1., 1.5, 2.]
    numpy.testing.assert_allclose(points[-1], [0., 0., 2.], atol=1.e-12)
    lines = lib_NICS.gaussian_bq_lines(points[:1])
    assert lines[0].split() == ['Bq', '0.000000', '0.000000', '0.000000']

def test_ring_normal_orientation():
    ring = [[numpy.cos(a), numpy.sin(a), 0.] for a in numpy.arange(6) * numpy.pi / 3]
    assert lib_NICS.ring_frame(ring)[1] == pytest.approx([0., 0., 1.])
    assert lib_NICS.ring_frame(ring[::-1])[1] == pytest.approx([0., 0., -1.])

def test_standard_orientation(tmp_path, ref):
    """
    Without NoSymm, Gaussian prints the tensors in the standard orientation.
    The Bq coordinates must be taken from the same frame.
    """
    R = rotation([1., -1., 0.], 1.1)
    shift = numpy.array([0.3, -1.2, 0.5])
    lines = open(LOG).readlines()
    ihead, start, end = orientation_block(lines)

    std = [lines[ihead].replace('Input orientation:', 'Standard orientation:')] + lines[ihead+1:start]
    for line in lines[start:end]:
        w = line.split()
        xyz = numpy.dot(R, numpy.array(w[3:6], float)) + shift
        std.append("%7s%11s%12s    %12.6f%12.6f%12.6f\n"%(w[0], w[1], w[2], xyz[0], xyz[1], xyz[2]))
    std.append(lines[end])

    out = lines[:end+1] + std
    i = end + 1
    while i < len(lines):
        out.append(lines[i])
        if 'Isotropic' in lines[i] and 'Anisotropy' in lines[i]:
            M = numpy.array([[float(x) for x in lines[i+k].split()[1::2]] for k in range(1, 4)])
            Mr = numpy.dot(R, numpy.dot(M, R.T))
            labs = [['XX', 'YX', 'ZX'], ['XY', 'YY', 'ZY'], ['XZ', 'YZ', 'ZZ']]
            for k in range(3):
                out.append(''.join('   %s=%11.4f'%(labs[k][j], Mr[k, j]) for j in range(3)) + '\n')
            i += 4
        else:
            i += 1
    nv = parse(write(tmp_path, 'standard.log', out))

    assert len(nv.NICS_data) == 8
    for p, q in zip(ref.NICS_data, nv.NICS_data):
        numpy.testing.assert_allclose(q.orig, numpy.dot(R, p.orig) + shift, atol=1.e-5)
        numpy.testing.assert_allclose(sorted(q.evals), sorted(p.evals), atol=2.e-3)
    numpy.testing.assert_allclose(nv.atoms[0]['coor'], numpy.dot(R, ref.atoms[0]['coor']) + shift, atol=1.e-5)

    res_ref = ref.ring_analysis(RING, outfile=None, lvprt=0)
    res_std = nv.ring_analysis(RING, outfile=None, lvprt=0)
    assert [r['label'] for r in res_std] == [r['label'] for r in res_ref]
    for a, b in zip(res_ref, res_std):
        assert b['height'] == pytest.approx(a['height'], abs=1.e-5)
        assert b['NICS_zz'] == pytest.approx(a['NICS_zz'], abs=2.e-3)

    # The test is sensitive: the ring normal of the input frame gives a different NICS_zz
    n_input = lib_NICS.ring_frame([ref.atoms[i-1]['coor'] for i in RING])[1]
    assert abs(nv.NICS_data[5].ret_NICS_comp(n_input) - res_ref[2]['NICS_zz']) > 1.

def test_freeze_codes_without_orientation(tmp_path, ref, capsys):
    lines = open(LOG).readlines()
    for i in zmat_bq_lines(lines):
        w = lines[i].split()
        lines[i] = " Bq  0  %s  %s  %s\n"%(w[1], w[2], w[3])
    ihead, start, end = orientation_block(lines)
    lines = lines[:ihead] + lines[end+1:]
    nv = parse(write(tmp_path, 'freeze.log', lines))
    assert 'no orientation table' in capsys.readouterr().out
    assert len(nv.NICS_data) == 8
    for p, q in zip(ref.NICS_data, nv.NICS_data):
        numpy.testing.assert_allclose(q.orig, p.orig, atol=1.e-5)
        assert q.NICS_iso == p.NICS_iso

def test_zmatrix_bq(tmp_path, ref):
    lines = open(LOG).readlines()
    for i in zmat_bq_lines(lines):
        lines[i] = " Bq                    1    B1       2    A1       3    D1\n"
    nv = parse(write(tmp_path, 'zmat.log', lines))
    for p, q in zip(ref.NICS_data, nv.NICS_data):
        numpy.testing.assert_allclose(q.orig, p.orig)

    # Without an orientation table the positions cannot be determined
    ihead, start, end = orientation_block(lines)
    path = write(tmp_path, 'zmat_noorient.log', lines[:ihead] + lines[end+1:])
    with pytest.raises(ValueError):
        parse(path)

def test_multistep(tmp_path, ref, capsys):
    lines = open(LOG).readlines()
    first = []
    for line in lines:
        if ' Bq   Isotropic' in line:
            line = line.replace('Isotropic =', 'Isotropic =   999.0000 ').split('Anisotropy')[0] + '\n'
        first.append(line)
    nv = parse(write(tmp_path, 'multistep.log', first + lines))
    assert 'using the last one' in capsys.readouterr().out
    assert len(nv.NICS_data) == 8
    for p, q in zip(ref.NICS_data, nv.NICS_data):
        assert q.NICS_iso == p.NICS_iso

def test_no_shielding(tmp_path):
    lines = [l for l in open(LOG).readlines() if not 'Magnetic shielding tensor' in l]
    with pytest.raises(ValueError):
        parse(write(tmp_path, 'noshield.log', lines))

def test_diag_output_independent_of_printoptions(ref, capsys):
    """Other modules change the global numpy print options."""
    point = ref.NICS_data[0] # complex eigenvalues -> diagnostic output
    point.diag()
    out_default = capsys.readouterr().out
    with numpy.printoptions(precision=3, suppress=True):
        point.diag()
    assert capsys.readouterr().out == out_default
    assert '0.70710678' in out_default

@pytest.mark.parametrize('qccode', ['gaussian', 'g09'])
def test_plot_vist_ring(tmp_path, qccode):
    os.chdir(tmp_path)
    with mock_stdout() as out:
        sys.argv = ['theodore', 'plot_vist', '-q', qccode, '-r', '4 5 7 9 8 6', LOG]
        ActionFactory.from_commandline()
    ring = open('NICS_ring.txt').read()
    assert 'NICS(0)' in ring and 'NICS(1)' in ring and 'NICS(-1)' in ring
    assert '-6.4439' in ring
    assert 'NICS_eig' in ring

def test_nics_prep_action(tmp_path, ref):
    os.chdir(tmp_path)
    with open('mol.xyz', 'w') as f:
        f.write("%i\n\n"%len(ref.atoms))
        for at in ref.atoms:
            f.write("%s %.6f %.6f %.6f\n"%({6: 'C', 1: 'H'}[at['Z']], *at['coor']))
    with mock_stdout() as out:
        sys.argv = ['theodore', 'nics_prep', 'mol.xyz', '4 5 7 9 8 6', '-b', '-q', 'gaussian', '-o', 'bq.txt']
        ActionFactory.from_commandline()
    bq = [l.split() for l in open('bq.txt').readlines()]
    assert len(bq) == 3 and all(w[0] == 'Bq' for w in bq)
    numpy.testing.assert_allclose([float(x) for x in bq[0][1:]], ref.NICS_data[3].orig, atol=1.e-5)

    # Other codes are not implemented yet
    with mock_stdout() as out:
        sys.argv = ['theodore', 'nics_prep', 'mol.xyz', '4 5 7 9 8 6', '-q', 'qchem']
        with pytest.raises(Exception, match='qccode'):
            ActionFactory.from_commandline()
