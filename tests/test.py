import pytest

def test_true():
    assert True

@pytest.mark.slow
def test_long():
    assert False

def test_slow():
    assert False
