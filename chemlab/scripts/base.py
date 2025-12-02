import os
import time

class Script:
    """
    Base class for all scripts used by CLI.

    Subclasses must define:
        name   : CLI subcommand name
        config : a ConfigBase subclass(If the script doesn't require config, config=None)
        run(self, cfg)
    """

    name = None
    config = None

    def run(self, cfg):
        raise NotImplementedError

class QchemBaseScript(Script):
    name = "Qchem"
    config = None
    def run(self, cfg):
        raise NotImplementedError
    @staticmethod
    def check_qchem_success(out_file):
        if not os.path.exists(out_file):
            return False
        with open(out_file) as f:
            return "Thank you very much for using Q-Chem" in f.read()

    def wait_for_jobs(self, out_files, log=None, interval=30):
        while True:
            if all(self.check_qchem_success(f) for f in out_files):
                msg = "All Q-Chem jobs completed."
                print(msg)
                if log: log.write(msg + "\n")
                return
            msg = "Waiting for Q-Chem jobs..."
            print(msg)
            if log: log.write(msg + "\n")
            time.sleep(interval)