import os
import sys
import numpy as np
import bitarray
import math
import pickle
import io
from typing import List, Any, Tuple
from PIL import Image
from codes.base import BaseCode
from tqdm import tqdm


class BaseChannel(object):
    def __init__(self, coder: BaseCode):
        super(BaseChannel, self).__init__()
        self.coder = coder
        self.block_size = coder.block_size
        self.code_size = coder.code_size
        self.int_num_blocks = math.ceil(65 / self.block_size)
        self.char_num_blocks = math.ceil(7 / self.block_size)

    def noise_block(self, x):
        assert x.shape[0] == self.code_size
        raise NotImplementedError()

    def transfer(self, message: Any, use_smart_pad=False) -> Any:
        f = io.BytesIO()
        pickle.dump(message, f)
        f.seek(0)
        x = bitarray.bitarray()
        x.fromfile(f)
        x = np.array(x.tolist(), dtype=np.int32)
        x = self.transfer_(x, use_smart_pad=use_smart_pad)
        x = bitarray.bitarray(''.join(map(str, (x % 2).tolist())))
        f = io.BytesIO(x.tobytes())
        f.seek(0)
        return pickle.load(f)

    def transfer_string(self, message: str, use_smart_pad=False) -> str:
        x = self.string2bit(message)
        x = self.transfer_(x, use_smart_pad=use_smart_pad)
        return self.bit2string(x)

    def transfer_int(self, message: int, use_smart_pad=False) -> int:
        x = self.int2bit(message)
        x = self.transfer_(x, use_smart_pad=use_smart_pad)
        return self.bit2int(x)

    def transfer_image(self, image: Image.Image):
        image = np.expand_dims(np.array(image), axis=-1)  # append pack dim
        image_bin = np.unpackbits(image, axis=-1)
        image_bin = np.reshape(image_bin, [image_bin.shape[0], image_bin.shape[1], -1])
        h, w, d = image_bin.shape

        image_bin = self.transfer_(image_bin.flatten(), use_smart_pad=False).reshape([h, w, d])

        image = np.packbits(image_bin.reshape([image_bin.shape[0], image_bin.shape[1], -1, 8]), axis=-1)
        image = np.squeeze(image, axis=-1)  # remove pack dim
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=-1)  # remove dim for gray image
        return Image.fromarray(image)

    def transfer_audio(self, audio):
        pass

    def transfer_(self, x: np.ndarray, use_smart_pad=True) -> np.ndarray:
        if use_smart_pad:
            x = self.pad_bits(x)
        else:
            x, l = self.pad_bits_simple(x)
        blocks = self.split_on_blocks(x)
        print("Encoding...")
        blocks = [self.coder.encode(block) for block in tqdm(blocks)]
        print("Transfering...")
        blocks = [self.noise_block(block) for block in tqdm(blocks)]
        print("Decoding...")
        blocks = [self.coder.decode(block) for block in tqdm(blocks)]
        x = self.merge_blocks(blocks)
        if use_smart_pad:
            x = self.unpad_bits(x)
        else:
            x = self.unpad_bits_simple(x, l)
        return x

    def split_on_blocks(self, x: np.ndarray) -> List[np.ndarray]:
        assert x.shape[0] % self.block_size == 0
        N = x.shape[0] // self.block_size
        blocks = np.split(x, N, axis=0)
        return blocks

    def merge_blocks(self, blocks: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(blocks, axis=0)

    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    def string2bit(self, s: str) -> np.ndarray:
        assert BaseChannel.is_ascii(s)

        x = bitarray.bitarray(''.join([bin(ord(x))[2:].zfill(7)
                                       for x in s]))
        x = np.array(x.tolist(), dtype=np.int32)
        return x

    def bit2string(self, x: np.ndarray) -> str:
        x = ''.join(map(str, (x % 2).tolist()))
        x = ''.join([chr(int(x[i: i + 7], 2))
                     for i in range(0, len(x), 7)])
        return x

    def int2bit(self, s: int) -> np.ndarray:
        assert s <= sys.maxsize
        real_bits = bin(s)[2:]
        x = real_bits.zfill(self.int_num_blocks * self.block_size)
        x = bitarray.bitarray(x)
        return np.array(x.tolist(), dtype=np.int32)

    def bit2int(self, x: np.ndarray) -> int:
        assert x.shape[0] == self.int_num_blocks * self.block_size
        x = ''.join(map(str, (x % 2).tolist()))
        x = int(x, 2)
        return x

    def pad_bits_simple(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        l = x.shape[0]
        red = (self.block_size - x.shape[0] % self.block_size) % self.block_size
        return np.concatenate([x, np.zeros(red, dtype=np.int32)], axis=0), l

    def unpad_bits_simple(self, x: np.ndarray, l: int) -> np.ndarray:
        return x[:l]

    def pad_bits(self, x: np.ndarray) -> np.ndarray:
        l = self.int2bit(x.shape[0])
        red = (self.block_size - x.shape[0] % self.block_size) % self.block_size
        return np.concatenate([l, x, np.zeros(red, dtype=np.int32)], axis=0)

    def unpad_bits(self, x: np.ndarray) -> np.ndarray:
        l = x[:self.block_size * self.int_num_blocks]
        l = self.bit2int(l)
        x = x[self.block_size * self.int_num_blocks: self.block_size * self.int_num_blocks + l]
        return x
