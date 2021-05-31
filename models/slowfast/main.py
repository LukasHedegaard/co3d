from ride import Main  # isort:skip
from models.slowfast.slowfast import SlowFastRide

if __name__ == "__main__":
    Main(SlowFastRide).argparse()
