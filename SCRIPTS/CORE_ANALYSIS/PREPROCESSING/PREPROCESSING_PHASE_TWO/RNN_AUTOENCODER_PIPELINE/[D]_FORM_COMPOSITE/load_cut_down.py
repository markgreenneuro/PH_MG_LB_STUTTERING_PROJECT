from pathlib import Path
import numpy as np
import pandas as pd

class LoadCutDown:
    def __init__(self, data_dir, primary_concrete_filename, *args, **kwargs):
        super(LoadCutDown, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.primary_concrete_filename = primary_concrete_filename
        self.primary_concrete_subtracted_filepath = Path.home().joinpath(self.data_dir,
                                                                         str(self.primary_concrete_filename))
        self.primary_concrete_subtracted = np.asarray(pd.read_csv(self.primary_concrete_subtracted_filepath))

    def return_cutdown(self):
        return self.primary_concrete_subtracted