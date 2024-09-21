from collections import Counter


def counter(text: str) -> list:
    count = Counter(text)
    total = sum(count.values())
    sorted_items = sorted(count.items(), key=lambda _: _[1], reverse=True)

    result = []
    previous_end = 0

    for char, num in sorted_items:
        percent = num / total
        result.append([char, [previous_end, previous_end + percent]])
        previous_end += percent

    return result


def build_graph_dict(graph: list) -> dict:
    return {g[0]: g[1] for g in graph}


def finder_encoder(text: str, graph_dict: dict) -> list:
    return graph_dict.get(text)


def encoder(text: str, graf: dict) -> float:
    new_graf = []
    worker = finder_encoder(text[0], graf)

    for i in range(1, len(text)):
        next_symbol = finder_encoder(text[i], graf)

        worker = [worker[0] + (worker[1] - worker[0]) * next_symbol[0],
                  worker[0] + (worker[1] - worker[0]) * next_symbol[1]]

        new_graf.append(worker)

        output_point = (new_graf[-1][1] - new_graf[-1][0]) + new_graf[-1][0]
    return output_point


def finder_decoder(point: float, table: list) -> list:
    for i in table:
        if i[1][0] <= point <= i[1][1]:
            return i


def decoder(counted: int, point: float, table: list) -> str:
    f_symbol = finder_decoder(point, table)
    result = [f_symbol[0]]

    for _ in range(0, counted - 1):
        point = ((point - f_symbol[1][0]) / (f_symbol[1][1] - f_symbol[1][0]))
        f_symbol = finder_decoder(point, table)
        result.append(f_symbol[0])

    return "".join(result)


if __name__ == '__main__':
    string = ""
    output = counter(string)
    graf_build = build_graph_dict(output)

    x = encoder(string, graf_build)
    decoded_string = decoder(len(string), x, output)
    print(decoded_string)

    if decoded_string == string:
        print("Success")
    else:
        print("Fail")
