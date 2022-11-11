#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import os

class CreateOutputDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateOutputDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.markov_chain = self.propagate_dir(results_dir, 'markov_chain')
        self.markov_chain_stats = self.propagate_dir(self.markov_chain, 'markov_chain_stats')
        self.markov_chain_models = self.propagate_dir(self.markov_chain, 'markov_chain_models')
        self.markov_chain_tensorboard = self.propagate_dir(self.markov_chain_models,
                                                                'markov_chain_tensorboard')
        self.markov_chain_partial_models = self.propagate_dir(self.markov_chain_models,
                                                                   'markov_chain_parameterised_part_models')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir
