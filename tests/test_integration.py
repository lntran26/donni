'''
Intergration test for the entire program
'''
import os
from subprocess import getstatusoutput, getoutput

PRG = '../dadinet/__main__.py'

def test_exists():
    """ Program exists """

    assert os.path.isfile(PRG)