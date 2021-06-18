import os
from contextlib import contextmanager


@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = os.getcwd()
    # print('Previous PATH:', prevdir)
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        # print('Switching back to previous PATH:', prevdir)
        os.chdir(prevdir)
