import sys
def main(filename):
    if "opt" in filename:
        sys.exit(1)
    elif "ene" in filename:
        sys.exit(2)
    elif "ts" in filename:
        sys.exit(3)
    elif "frez" in filename:
        sys.exit(4)
    else:
        sys.exit(0)
main(sys.argv[1])