from typing import List, Set, Tuple

import numpy as np


class Position:
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def __init__(self, x: int, y: int, dir: int):
        self.x = x
        self.y = y
        self.dir = dir

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.dir == other.dir

    def __hash__(self):
        return hash((self.x, self.y, self.dir))

    def turn_left(self):
        self.dir = (self.dir - 1) % 4

    def turn_right(self):
        self.dir = (self.dir + 1) % 4

    @property
    def xy(self) -> Tuple[int, int]:
        return self.x, self.y

    def step(self):
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.dir]
        self.x += dx
        self.y += dy

    def left_pixel(self) -> Tuple[int, int]:
        dx, dy = [(0, -1), (0, 0), (-1, 0), (-1, -1)][self.dir]
        return self.x + dx, self.y + dy

    def right_pixel(self) -> Tuple[int, int]:
        dx, dy = [(0, 0), (-1, 0), (-1, -1), (0, -1)][self.dir]
        return self.x + dx, self.y + dy


def trace_contours(img: np.ndarray) -> List[np.ndarray]:
    '''Trace the contours of a binary image.

    This function uses a simple edge-following algorithm to trace the contours.

    Args:
        img: 2D numpy array where zero pixels represent the background.

    Returns:
        contours: List of 2D numpy arrays with the contours.
    '''
    contours = []
    contour_lengths = []
    pad = 1
    img = np.pad(img, 1, mode='constant')
    h, w = img.shape
    visited = set()

    for y in range(h - 1):
        for x in range(w):
            if not img[y, x] and img[y + 1, x]:
                start = Position(x, y + 1, Position.RIGHT)
                if start not in visited:
                    num_visited = len(visited)
                    contour = trace_single_contour(img, visited, start)
                    contours.append(contour - pad)
                    contour_lengths.append(len(visited) - num_visited)

    contours = [contours[i] - 0.5 for i in np.argsort(contour_lengths)[::-1]]
    return contours


def trace_single_contour(img: np.ndarray, visited: Set, start: Position) -> np.ndarray:
    position = Position(start.x, start.y, start.dir)
    contour = [position.xy]
    visited.add(position)

    position.step()
    visited.add(position)

    while position.xy != start.xy:
        left, right = position.left_pixel(), position.right_pixel()

        if img[left[1], left[0]] and img[right[1], right[0]]:
            position.turn_left()
            contour.append(position.xy)
        elif (not img[left[1], left[0]] and not img[right[1], right[0]]) or (
            img[left[1], left[0]] and not img[right[1], right[0]]
        ):
            position.turn_right()
            contour.append(position.xy)

        position.step()
        visited.add(position)

    return np.array(contour)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.random.rand(16, 16) > 0.4
    contours = trace_contours(img)

    plt.imshow(img, cmap='gray')
    for contour in contours:
        pts = np.vstack([contour, contour[0]])
        plt.plot(pts[:, 0], pts[:, 1], linewidth=3)
    plt.show()
