class Script:
    """
    Base class for all scripts used by CLI.

    Subclasses must define:
        name   : CLI subcommand name
        config : a ConfigBase subclass
        run(self, cfg)
    """

    name = None
    config = None

    def run(self, cfg):
        raise NotImplementedError
