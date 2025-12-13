
"""
Module to expose more detailed version info for the installed `scipy`
"""
version = "1.16.3"
full_version = version
short_version = version.split('.dev')[0]
git_revision = "b9105ccc2237f57acb1060202cd77f6dd264fb34"
release = 'dev' not in version and '+' not in version

if not release:
    version = full_version
