from re import match
from pathlib import Path
from shutil import move, make_archive, rmtree
from os import getcwd, chdir, path, mkdir, listdir

from contextlib import contextmanager


@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = getcwd()
    # print('Previous PATH:', prevdir)
    chdir(path.expanduser(newdir))
    try:
        yield
    finally:
        # print('Switching back to previous PATH:', prevdir)
        chdir(prevdir)


def zip_run_name_files(run_name):

    # Make sure a output directory exist
    if not path.exists('output'):
        mkdir('output')

    # The directory where files are now
    current_dir = Path(getcwd()).absolute()

    # The directory to put files into
    working_dir = current_dir / 'output' / run_name
    mkdir(working_dir)

    # Move all files from current dir to working dir
    for f in listdir():
        if match(run_name, f):
            move(current_dir / f, working_dir / f)

    # Zip the new directory
    make_archive(working_dir, 'zip', working_dir)

    # Delete the non-zipped directory
    rmtree(working_dir)
