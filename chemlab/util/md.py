import numpy as np
import re

class traj(object):
    def __init__(self):
        self.frames = []

    @property
    def _nframes(self):
        return len(self.frames)

    def read_xyz_traj(self, filename, type="vmd"):
        file = open(filename, "r").readlines()
        if type == "vmd":
            total_atoms = int(file[0])
            i = 2
            while i < len(file) - 1:
                lines = []
                for j in range(total_atoms):
                    line = file[i + j].split()
                    if len(line) != 4:
                        print(f"Error in line {i+j}:{file[i+j]}")
                    line[0] = re.sub(r'\d+', '', line[0])
                    lines.append(line)
                self.frames.append(np.asarray(lines))
                i += total_atoms + 2
        else:
            raise TypeError(f"Invalid type: {type}")