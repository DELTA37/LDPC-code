import argparse
from noise_channel.straight import StraightChannel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    channel = StraightChannel(12)
    print(channel.transfer(args.message))
