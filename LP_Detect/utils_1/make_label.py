import numpy


def point_in_polygon(x, y, pts):
    n = len(pts) // 2
    pts_x = [pts[i] for i in range(0, n)]
    pts_y = [pts[i] for i in range(n, len(pts))]
    if not min(pts_x) <= x <= max(pts_x) or not min(pts_y) <= y <= max(pts_y):
        return False
    res = False
    for i in range(n):
        j = n - 1 if i == 0 else i - 1
        if ((pts_y[i] > y) != (pts_y[j] > y)) and (x < (pts_x[j] - pts_x[i]) * (y - pts_y[i]) / (pts_y[j] - pts_y[i]) + pts_x[i]):
            res = not res
    return res


def object_label(points, dims, stride):
    scale = ((dims + 40.0) / 2.0) / stride
    size = dims // stride
    label = numpy.zeros((size, size, 9))
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            if point_in_polygon(x, y, points):
                label[i, j, 0] = 1
                pts = numpy.array(points).reshape((2, -1))
                pts = pts * dims / stride
                pts = pts - numpy.array([[j + 0.5], [i + 0.5]])
                pts = pts / scale
                label[i, j, 1:] = pts.reshape((-1,))
    return label