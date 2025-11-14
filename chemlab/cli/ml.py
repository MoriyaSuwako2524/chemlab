from chemlab.cli.base import CLICommand

class MlData(CLICommand):
    name = "ml_data"

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest="ml_cmd")

        train = sub.add_parser("train")
        train.set_defaults(func=self.train)

        eval_ = sub.add_parser("eval")
        eval_.set_defaults(func=self.eval)

    def train(self, args):
        print("[ML] Training...")

    def eval(self, args):
        print("[ML] Evaluation...")
