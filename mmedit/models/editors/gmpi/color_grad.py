# modified from https://bsouthga.dev/posts/color-gradients-with-python

import numpy as np


def hex_to_RGB(hex):
    """ "#FFFFFF" -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    '''[255,255,255] -> "#FFFFFF"'''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB])


def rand_hex_color(num=1):
    """Generate random hex colors, default is one,
    returning a string. If num is greater than
    1, an array of strings is returned."""
    colors = [RGB_to_hex([x * 255 for x in np.random.rand(3)]) for i in range(num)]
    if num == 1:
        return colors[0]
    else:
        return colors


def color_dict(gradient):
    """Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on"""
    return {
        "hex": [RGB_to_hex(RGB) for RGB in gradient],
        "r": [RGB[0] for RGB in gradient],
        "g": [RGB[1] for RGB in gradient],
        "b": [RGB[2] for RGB in gradient],
    }


def rgb_from_color_dict(dict_info):
    rgb = np.array([dict_info["r"], dict_info["g"], dict_info["b"]])
    return rgb


# ----------------------------------------------------------
# linear gradient and linear interpolation


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF")"""
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)


# ------------------------------------------------------------------------
# multiple linear gradient, polylinear interpolation


def polylinear_gradient(colors, n):
    """returns a list of colors forming linear gradients between
    all sequential pairs of colors. "n" specifies the total
    number of desired output colors"""
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col + 1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict


# -----------------------------------------------------------------------
# nonlinear gradient, Bezier interpolation

# Value cache
FACTORIAL_CACHE = {}


def fact(n):
    """Memoized factorial function"""
    try:
        return FACTORIAL_CACHE[n]
    except (KeyError):
        if n == 1 or n == 0:
            result = 1
        else:
            result = n * fact(n - 1)
        FACTORIAL_CACHE[n] = result
        return result


def bernstein(t, n, i):
    """Bernstein coefficient"""
    binom = fact(n) / float(fact(i) * fact(n - i))
    return binom * ((1 - t) ** (n - i)) * (t**i)


def bezier_gradient(colors, n_out=100):
    """Returns a "bezier gradient" dictionary
    using a given list of colors as control
    points. Dictionary also contains control
    colors/points."""
    # RGB vectors for each color, use as control points
    RGB_list = [hex_to_RGB(color) for color in colors]
    n = len(RGB_list) - 1

    def bezier_interp(t):
        """Define an interpolation function
        for this specific curve"""
        # List of all summands
        summands = [list(map(lambda x: int(bernstein(t, n, i) * x), c)) for i, c in enumerate(RGB_list)]
        # Output color
        out = [0, 0, 0]
        # Add components of each summand together
        for vector in summands:
            for c in range(3):
                out[c] += vector[c]

        return out

    gradient = [bezier_interp(float(t) / (n_out - 1)) for t in range(n_out)]
    # Return all points requested for gradient
    return {"gradient": color_dict(gradient), "control": color_dict(RGB_list)}
