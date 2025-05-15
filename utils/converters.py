import base64

def ToBase64(path: str) -> str:
    """
    Convert a file to a base64 string.

    Args:
        path (str): The path to the file.

    Returns:
        str: The base64 string of the file.
    """

    with open(path, "rb") as file:
        mimetype = path.split(".")[-1]
        base = base64.b64encode(file.read()).decode("utf-8")
        return f"data:image/{mimetype};base64,{base}"


def base64_to_file(data_uri: str, output_path: str):
    _, encoded = data_uri.split(",", 1)
    binary_data = base64.b64decode(encoded)

    with open(output_path, "wb") as f:
        f.write(binary_data)
