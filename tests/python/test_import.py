import specklebem as sb


def test_version():
    assert sb.__version__


def test_reference_materials():
    ag = sb.silver_500nm()
    assert ag.eps_r.real < 0
    assert ag.n.imag <= 0
    kind, precond = sb.recommend_formulation(ag.eps_r)
    assert kind == sb.Formulation.ICTF
    assert precond is True
