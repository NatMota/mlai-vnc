"""Quote conda Clang's generated sysroot argument when the environment path has spaces."""
import os
from pathlib import Path

prefix = Path(os.environ['CONDA_PREFIX'])
config = prefix / 'bin/x86_64-conda-linux-gnu.cfg'
if config.exists() and ' ' in str(prefix):
    original = config.read_text()
    argument = '--sysroot=' + str(prefix / 'x86_64-conda-linux-gnu/sysroot')
    corrected = original.replace(argument + '\n', '--sysroot="' + argument.split('=', 1)[1] + '"\n')
    if corrected != original:
        temporary = config.with_suffix('.cfg.quoted')
        temporary.write_text(corrected)
        temporary.replace(config)  # Do not alter a potentially hard-linked package cache file.
