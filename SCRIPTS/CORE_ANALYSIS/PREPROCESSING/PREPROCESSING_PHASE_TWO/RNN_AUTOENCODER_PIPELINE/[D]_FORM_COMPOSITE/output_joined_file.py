from pathlib import Path
import pandas as pd

class ProduceOutputJointDataSet:
    def __init__(self, data_dir, joined_cvae_zlatent_file, joined_cvae_zlatent_filename, *args, **kwargs):
        super(ProduceOutputJointDataSet, self).__init__(*args, **kwargs)
        self.data_dir = Path(data_dir)
        self.joined_cave_zlatent_file = joined_cvae_zlatent_file
        self.joined_cvae_zlatent_filename = joined_cvae_zlatent_filename
        os.chdir(data_dir)
        self.joined_cave_zlatent_filepath = Path.home().joinpath(self.data_dir,
                                                                 str(self.joined_cvae_zlatent_filename))
        self.output_joint_dataset()

    def output_joint_dataset(self):
        return pd.DataFrame(self.joined_cave_zlatent_file).to_csv(self.joined_cave_zlatent_filepath)
