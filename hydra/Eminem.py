import pathlib
"""
Read and Write a "WIndowTracker" file. 
WindowTracker has window and last run columns
On run, read WindowTracker, create new "Environment" tasks for each window that needs to be rerun
When running a window, all subwindows are removed and only replaced if the relevant wavelength is still significant
If Root Window needs to be rerun, erase all subwindows and start Environment task for that window
    Create Environment Tasks for all significant frequency/wavelengths, recursively
    Update WindowTracker file

"""

def main():
    outputdor = pathlib.Path('./output/')

    with open(filename, "wb") as handle:

if __name__ == '__main__':
    main()