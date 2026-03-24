from __future__ import annotations

from typing import Any, Literal, overload

import s3fs


@overload
def find(  # type: ignore
    s3: s3fs.S3FileSystem,
    path: str,
    maxdepth: int | None = None,
    withdirs: bool = False,
    detail: Literal[False] = ...,
) -> list[str]:
    pass


@overload
def find(
    s3: s3fs.S3FileSystem,
    path: str,
    maxdepth: int | None = None,
    withdirs: bool = False,
    detail: Literal[True] = ...,
) -> dict[str, dict[str, Any]]:
    pass


def find(
    s3: s3fs.S3FileSystem,
    path: str,
    maxdepth: int | None = None,
    withdirs: bool = False,
    detail: bool = False,
) -> list[str] | dict[str, dict[str, Any]]:
    """List all files below path.

    This function is exactly equivalent to s3fs's find() function, but much
    faster because it relies on the underlying boto3 client's pagination instead
    of recursively listing each subdirectory.

    Args:
        s3: An s3fs filesystem instance (used to resolve endpoint / region).
        path: The path to search for files. Must point to a directory.
        maxdepth: The maximum number of levels to descend. The given path itself
            is depth 0, items directly inside it are depth 1, items inside a
            subdirectory are depth 2, etc. If None, there is no limit.
        withdirs: Whether to include directory paths in the output. If False,
            only object paths are included.
            Warning: explicit directory marker objects (keys that end with "/"
            and have zero size) are still included if withdirs is False! These
            markers can essentially be seen as zero-byte files whose name is
            an empty string.
        detail: Whether to return a dict with file details instead of just
            file paths. If True, a dict with file details is returned.

    Returns:
        A list of "bucket/key" strings sorted in the order they are encountered
        if detail is False, or a dict mapping "bucket/key" strings to dicts
        containing additional metadata if detail is True.
    """
    if not s3.isdir(path):
        raise ValueError(f"S3 path '{path}' is not a directory.")

    if maxdepth is not None and maxdepth < 0:
        raise ValueError(
            f"maxdepth must be non-negative or None, got {maxdepth}."
        )

    # Normalize the path and split into bucket + key prefix.
    root_path = path.removeprefix("s3://").removesuffix("/")
    bucket, _, prefix = root_path.partition("/")

    # Ensure the prefix ends with "/" so we don't accidentally match a sibling
    # directory that shares the same leading characters.
    if prefix:
        prefix += "/"

    result_list = []
    result_details = {}
    seen_dirs = {()}  # always add the root

    # List all objects, reusing the already configured boto3 client.
    paginator = s3.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_key = key[len(prefix) :]
            file_path = f"{bucket}/{key}"

            # Split into path components.
            rel_key_parts = rel_key.split("/")
            depth = len(rel_key_parts)

            # Add the file to the results if it's within maxdepth.
            if maxdepth is None or depth <= maxdepth:
                if detail:
                    # For some weird reason, s3fs's find() with detail=True
                    # returns a dict with all sorts of extra duplicate values.
                    # We just mirror that behavior here for consistency.
                    obj["Key"] = file_path
                    obj["type"] = "file"
                    obj["size"] = obj["Size"]
                    obj["name"] = file_path
                    result_details[file_path] = obj
                else:
                    result_list.append(file_path)

            if not withdirs:
                continue

            # Collect intermediate virtual directory paths when requested.
            for i in range(
                1, min(depth, maxdepth + 1) if maxdepth is not None else depth
            ):
                seen_dirs.add(tuple(rel_key_parts[:i]))

    if withdirs:
        # Add the seen directories to the results.
        for dir_parts in seen_dirs:
            dir_path = "/".join((root_path,) + dir_parts)
            if detail:
                result_details[dir_path] = {
                    "Key": dir_path,
                    "Size": 0,
                    "StorageClass": "DIRECTORY",
                    "type": "directory",
                    "size": 0,
                    "name": dir_path,
                }
            else:
                result_list.append(dir_path)

    return result_details if detail else result_list
