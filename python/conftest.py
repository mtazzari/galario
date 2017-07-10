import pytest

def pytest_addoption(parser):
    parser.addoption("--gpu", action="store", default=0,
        help="Run tests on gpu. Default: 0")
