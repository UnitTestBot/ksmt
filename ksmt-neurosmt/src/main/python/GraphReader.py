from enum import Enum
from typing import Union, TextIO


class VertexType(Enum):
    SYMBOLIC = 0
    VALUE = 1
    APP = 2


def get_vertex_type(name: str):
    if name == "SYMBOLIC":
        return VertexType.SYMBOLIC

    if name == "VALUE":
        return VertexType.VALUE

    return VertexType.APP


def process_line(line: str, cur_vertex: int, operators: list[str], edges: list[tuple[int, int]], depth: list[int]):
    name, info = line.split(";")
    info = info.strip()
    # depth[v] is a length of the longest path from vertex v to any sink in an expression graph
    depth.append(0)

    if get_vertex_type(name) == VertexType.APP:
        operators.append(name)

        children = map(int, info.split(" "))
        for u in children:
            edges.append((u, cur_vertex))
            depth[cur_vertex] = max(depth[cur_vertex], depth[u] + 1)

    else:
        operators.append(name + ";" + info)


def read_graph_from_file(inf: TextIO, max_size: int, max_depth: int)\
        -> Union[tuple[list[str], list[tuple[int, int]], list[int]], tuple[None, None, None]]:

    operators, edges, depth = [], [], []

    cur_vertex = 0
    for line in inf.readlines():
        line = line.strip()

        if line.startswith(";"):
            continue  # lines starting with ";" are considered to be comments

        try:
            process_line(line, cur_vertex, operators, edges, depth)

        except Exception as e:
            print(e, "\n")
            print(inf.name, "\n")
            print(cur_vertex, line, "\n")
            raise e

        if cur_vertex >= max_size or depth[cur_vertex] > max_depth:
            return None, None, None

        cur_vertex += 1

    return operators, edges, depth


def read_graph_by_path(path: str, max_size: int, max_depth: int)\
        -> Union[tuple[list[str], list[tuple[int, int]], list[int]], tuple[None, None, None]]:

    with open(path, "r") as inf:
        try:
            return read_graph_from_file(inf, max_size, max_depth)

        except Exception as e:
            print(e)
            print(f"path: '{path}'")

            raise e
