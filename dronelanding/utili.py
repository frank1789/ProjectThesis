#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+hlen//3], 16)/255 for i in range(0, hlen, hlen//3))



if __name__ == "__main__":
    print(hex_to_rgb("#ffffff"))
    print(hex_to_rgb("#b0ecff"))
    print(hex_to_rgb("#fbeaea"))
    print(hex_to_rgb("#bcbcbc"))

