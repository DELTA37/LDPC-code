from codes.poly_gf2 import poly1d_gf2


if __name__ == '__main__':
    a = poly1d_gf2([1, 0, 1, 0, 1, 0, 1, 1])
    b = poly1d_gf2([1, 0, 1, 1])
    q, r = a.euclid_div(b)
    print(f"a: {repr(a)} = q: {repr(q)} * b: {repr(b)} + r: {repr(r)}")
