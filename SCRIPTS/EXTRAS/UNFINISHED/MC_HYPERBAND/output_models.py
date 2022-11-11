import os


class OutputModel:
    def __init__(self, markov_chain_dir, run_tuner_get_best_hyperparameters, *args, **kwargs):
        super(OutputModel, self).__init__(*args, **kwargs)
        self.set_model_output(markov_chain_dir)
        self.save_model(run_tuner_get_best_hyperparameters)
        self.markov_chain_models = markov_chain_dir.markov_chain_models

    def save_model(self, run_tuner_get_best_hyperparameters):
        os.chdir(self.markov_chain_models)
        self.model = run_tuner_get_best_hyperparameters
        self.model.save('markov_chain_winning.model')

