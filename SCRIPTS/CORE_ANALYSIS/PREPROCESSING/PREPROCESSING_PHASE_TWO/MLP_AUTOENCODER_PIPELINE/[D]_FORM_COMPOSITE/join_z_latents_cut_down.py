import numpy as np
from load_cut_down import *
from load_z_latents import *


class JoinZlatentsCutDown:
    def __init__(self, data_dir, latent_filename, latent_modality, latent_suffix,
                 primary_concrete_filename,
                 *args, **kwargs):
        super(JoinZlatentsCutDown, self).__init__(*args, **kwargs)
        self.primary_concrete_subtracted = None
        self.zlatents = None
        self.data_dir = data_dir
        self.latent_filename = latent_filename
        self.latent_modality = latent_modality
        self.latent_suffix = latent_suffix
        self.primary_concrete_filename = primary_concrete_filename

    def join(self):
        self.primary_concrete_subtracted = LoadCutDown(data_dir=self.data_dir,
                                                       primary_concrete_filename=
                                                       self.primary_concrete_filename).return_cutdown()

        self.zlatents = LoadOutputLatents(data_dir=self.data_dir,
                                          latent_filename=self.latent_filename,
                                          latent_modality=self.latent_modality,
                                          latent_suffix=self.latent_suffix).return_latents()

        return np.hstack((self.primary_concrete_subtracted, self.zlatents))
