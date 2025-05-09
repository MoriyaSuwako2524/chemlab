import sys
def main(filename):
    if "tem1" in filename:
        sys.exit(1)
    elif "tem2" in filename:
        sys.exit(2)
    elif "tem3" in filename:
        sys.exit(3)
    else:
        sys.exit(0)
main(sys.argv[1])