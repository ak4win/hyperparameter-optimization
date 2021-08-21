import os
import json


class JsonResultsSaver:
    def __init__(self, directory, overwrite=False):
        """
        As the logger class provided by the package does not fullfil our needs,
        we create our own one that has the same interface and pass it to the
        package's classes.
        """
        # Ensure the given directory exists
        os.makedirs(directory, exist_ok=True)
        self.results_path = os.path.join(directory, 'results.json')

        # Ensure the config file exists/is reset
        try:
            # Operator x is exclusively for file creation, throws FileExistsError if file exists already
            with open(self.results_path, 'x') as _:
                pass
        except FileExistsError:
            if overwrite:
                # This will open the file and put empty content in it
                with open(self.results_path, 'w') as file:
                    file.write('')
            else:
                raise FileExistsError(f'The file {self.results_path} already exists.')
        except Exception as e:
            raise e

    def new_config(self, config_id, config, config_info):
        pass
        # if config_id not in self.config_ids:
        #     self.config_ids.add(config_id)
        #     with open(self.config_fn, 'a') as fh:
        #         fh.write(json.dumps([config_id, config, config_info]))
        #         fh.write('\n')

    def __call__(self, job):
        with open(self.results_path, 'a') as file:
            json.dump(
                {
                    "id": job.id,
                    "budget": job.kwargs['budget'],
                    "loss": job.result['loss'],
                    "config": job.kwargs['config'],
                    "result": job.result,
                    "timestamp": job.timestamps,
                    "exception": job.exception
                },
                file
            )
            file.write("\n")
