from pathlib import Path
import numpy as np
import pandas as pd

class LoadOutputLatents:
    def __init__(self, data_dir, latent_filename, latent_modality, latent_suffix, *args, **kwargs):
        super(LoadOutputLatents, self).__init__(*args, **kwargs)
        self.latent_filepath = None
        self.latent_filename_full = None
        self.data_dir = data_dir
        self.latent_filename = latent_filename
        self.latent_modality = latent_modality
        self.latent_suffix = latent_suffix
        self.assemble_full_filename()
        self.create_path_to_output()

    def assemble_full_filename(self):
        self.latent_filename_full = str(self.latent_filename) + '_' + \
                                    str(self.latent_modality) + '_' + \
                                    str(self.latent_suffix)
        return self

    def create_path_to_output(self):
        self.latent_filepath = Path.home().joinpath(self.data_dir, str(self.latent_filename_full))
        return self

    def return_latents(self):
        zlatents = np.asarray(pd.read_csv(self.latent_filepath))
        return zlatents