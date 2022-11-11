from typing_extensions import Self

class Vec2:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def within(self, a: Self, b: Self):
        return (
            (self.x >= a.x and self.x <= b.x)
            and
            (self.y >= a.x and self.y <= b.x)
        )

    def __eq__(self, o: Self):
        return self.x == o.x and self.y == o.y

    def __add__(self, o: Self):
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: Self):
        return Vec2(self.x - o.x, self.y - o.y)

    def __str__(self):
        return f"Vec2({self.x}, {self.y})"

    def __repr__(self):
        return f"vec.Vec2({self.x.__repr__()}, {self.y.__repr__()})"

    def manhattan_dist(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)
