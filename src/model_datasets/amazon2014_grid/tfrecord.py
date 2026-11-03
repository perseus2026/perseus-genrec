"""Read GRID's gzip TFRecord/tf.train.Example without importing TensorFlow.

Wire schema follows tensorflow/core/example/{example,feature}.proto. Existing
protobuf is reused. Gzip CRC and record framing are checked; TFRecord CRC32C
fields are consumed but not verified. This is explicitly reported by the import.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _example_type() -> Any:
    schema = descriptor_pb2.FileDescriptorProto(
        name="grid_example.proto", package="grid_import", syntax="proto3"
    )

    def add_field(
        message: Any,
        name: str,
        number: int,
        kind: int,
        *,
        repeated: bool = False,
        type_name: str | None = None,
        oneof: int | None = None,
    ) -> None:
        value = message.field.add(
            name=name, number=number, type=kind, label=3 if repeated else 1
        )
        if type_name is not None:
            value.type_name = f".grid_import.{type_name}"
        if oneof is not None:
            value.oneof_index = oneof

    for name, kind in (("BytesList", 12), ("FloatList", 2), ("Int64List", 3)):
        message = schema.message_type.add(name=name)
        add_field(message, "value", 1, kind, repeated=True)
    feature = schema.message_type.add(name="Feature")
    feature.oneof_decl.add(name="kind")
    for number, (name, type_name) in enumerate(
        (
            ("bytes_list", "BytesList"),
            ("float_list", "FloatList"),
            ("int64_list", "Int64List"),
        ),
        start=1,
    ):
        add_field(feature, name, number, 11, type_name=type_name, oneof=0)
    features = schema.message_type.add(name="Features")
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    add_field(entry, "key", 1, 9)
    add_field(entry, "value", 2, 11, type_name="Feature")
    add_field(
        features, "feature", 1, 11, repeated=True, type_name="Features.FeatureEntry"
    )
    example = schema.message_type.add(name="Example")
    add_field(example, "features", 1, 11, type_name="Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("grid_import.Example")
    )


_EXAMPLE = _example_type()


def records(path: Path) -> Iterator[dict[str, tuple[str, Any]]]:
    """Yield typed feature lists from one complete compressed record stream."""
    with gzip.open(path, "rb") as stream:
        index = 0
        while header := stream.read(12):
            index += 1
            if len(header) != 12:
                raise ValueError(f"{path}: truncated TFRecord header at record {index}")
            length = struct.unpack("<Q", header[:8])[0]
            if not 0 < length <= 128 * 1024 * 1024:
                raise ValueError(f"{path}: invalid record length {length}")
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ValueError(
                    f"{path}: truncated TFRecord payload at record {index}"
                )
            example = _EXAMPLE()
            example.ParseFromString(payload)
            values = {}
            for name, feature in example.features.feature.items():
                kind = feature.WhichOneof("kind")
                if kind is None:
                    raise ValueError(f"{path}: empty feature {name!r}")
                values[name] = (kind, getattr(feature, kind).value)
            yield values
