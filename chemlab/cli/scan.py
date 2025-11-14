from chemlab.cli.base import CLICommand

class Scan(CLICommand):
    name = "scan"

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest="scan_cmd")

        run = sub.add_parser("run")
        run.set_defaults(func=self.run_cmd)

    def run_cmd(self, args):
        print("[SCAN] Running scan...")
