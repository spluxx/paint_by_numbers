from pathlib import Path
from skimage import io
from pbn import pbn, export_result


def main():
    Path("./out").mkdir(parents=True, exist_ok=True)

    image_paths = [
        ("girl", "images/girl.jpeg"),
        ("kangaroo", "images/kangaroo.jpg"),
        ("monet", "images/monet.jpg"),
        ("monet2", "images/monet2.png"),
        ("turtle", "images/turtle.jpeg"),
    ]

    num_colors = [8, 16, 24]

    smoothing_factors = [0.2, 0.5]

    for C in num_colors:
        for name, path in image_paths:
            for s in smoothing_factors:
                I = io.imread(path)[:, :, :3]
                colored, contour = pbn(I, C, s)
                export_result(
                    "./out", "{}_{}_{}".format(name, C, str(s).replace(".", "_")), colored, contour)


if __name__ == '__main__':
    main()
