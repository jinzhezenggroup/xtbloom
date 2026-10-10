"""Pure-Python tests for bounded mapped-image endpoint observations."""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import runtime_mapped_images as mapped_images


class ParseMapsTest(unittest.TestCase):
    """Exercise maps parsing without consulting the current process."""

    def test_kernel_column_padding_is_not_part_of_the_absolute_path(self) -> None:
        """Keep the kernel's alignment separate from spaces inside a filename."""
        text = (
            "00400000-00420000 r--p 00000000 103:02 97017335"
            "                          /tmp/lib with space].so\n"
            "00500000-00520000 r-xp 00000000 103:02 97017336"
            "\t\t/tmp/ends-in-bracket]\n"
            "00600000-00620000 rw-p 00000000 00:00 0                  \n"
        )
        mappings = mapped_images.parse_maps(text)
        self.assertEqual(mappings[0]["path"], "/tmp/lib with space].so")
        self.assertEqual(mappings[1]["path"], "/tmp/ends-in-bracket]")
        self.assertEqual(mappings[2]["kind"], "anonymous")

    def test_parses_file_paths_with_spaces_unicode_and_non_file_mappings(self) -> None:
        """Keep pathname content distinct from kernel mapping labels."""
        text = (
            "1000-2000 r-xp 00000000 08:01 123 /tmp/lib with space Ω.so\n"
            "3000-4000 rw-p 00000000 00:00 0\n"
            "5000-6000 rw-p 00000000 00:00 0 [heap]\n"
        )
        mappings = mapped_images.parse_maps(text)
        self.assertEqual(mappings[0]["path"], "/tmp/lib with space Ω.so")
        self.assertEqual(mappings[0]["kind"], "file")
        self.assertEqual(mappings[0]["permissions"], "r-xp")
        self.assertEqual(mappings[0]["offset"], 0)
        self.assertEqual(mappings[0]["device_major"], 8)
        self.assertEqual(mappings[0]["device_minor"], 1)
        self.assertEqual(mappings[0]["inode"], 123)
        self.assertEqual(mappings[1]["kind"], "anonymous")
        self.assertEqual(mappings[2]["kind"], "pseudopath")

    def test_rejects_malformed_ranges_tags_unicode_and_ambiguous_paths(self) -> None:
        """Reject records that cannot identify a backing file unambiguously."""
        invalid = (
            "1000-1000 r-xp 0 08:01 1 /tmp/a\n",
            "2000-1000 r-xp 0 08:01 1 /tmp/a\n",
            "1000-10000000000000000 r-xp 0 08:01 1 /tmp/a\n",
            "1000-2000 rwqp 0 08:01 1 /tmp/a\n",
            "1000-2000 r-xp nope 08:01 1 /tmp/a\n",
            "1000-2000 r-xp 0 08:zz 1 /tmp/a\n",
            "1000-2000 r-xp 0 08:01 -1 /tmp/a\n",
            "1000-2000 r-xp 0 08:01 1 /tmp/a",
            "1000-2000 r-xp 0 08:01 1 /tmp/a\n\n",
            "1000-2000 r-xp 0 08:01 1 /tmp/a\\040b\n",
            "1000-2000 r-xp 0 08:01 1 /tmp/a (deleted)\n",
            "1000-2000 r-xp 0 08:01 1 relative/path\n",
            "1000-2000 r-xp 0 08:01 1 [heap\n",
            "1000-2000 r-xp 0 08:01 1 /tmp/bad\udcff\n",
            "1000-2000 r-xp 0 08:01 1 /tmp/with\ttab\n",
        )
        for text in invalid:
            with (
                self.subTest(text=repr(text)),
                self.assertRaises(mapped_images.MappingError),
            ):
                mapped_images.parse_maps(text)

    def test_rejects_overlapping_and_duplicate_ranges(self) -> None:
        """An address cannot belong to two candidate mappings."""
        overlap = "1000-2000 r-xp 0 08:01 1 /tmp/a\n1800-2800 r--p 0 08:01 1 /tmp/a\n"
        duplicate = "1000-2000 r-xp 0 08:01 1 /tmp/a\n" * 2
        for text in (overlap, duplicate):
            with self.subTest(text=text), self.assertRaises(mapped_images.MappingError):
                mapped_images.parse_maps(text)

    def test_accepts_empty_snapshot_as_no_mappings(self) -> None:
        """An empty parse does not invent an image observation."""
        self.assertEqual(mapped_images.parse_maps(""), [])


class RuntimeMappedImagesTest(unittest.TestCase):
    """Verify capture pins mapped files without loading or calling them."""

    def setUp(self) -> None:
        """Use synthetic maps and ELF-like bytes, never a loaded library."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library_path = self.root / "lib xtbloom 测试.so"
        self.library_path.write_bytes(b"\x7fELF" + bytes(range(32)))
        self.maps_text = self._map_row(self.library_path)
        self.entrypoints = {
            "owner-0:xtbloom_compute": 0x1010,
            "owner-0:xtbloom_context_create": 0x1020,
        }

    @staticmethod
    def _map_row(
        path: Path,
        *,
        start: int = 0x1000,
        end: int = 0x2000,
        permissions: str = "r-xp",
        device: tuple[int, int] | None = None,
        inode: int | None = None,
        mapped_path: str | None = None,
    ) -> str:
        metadata = path.stat()
        device_major, device_minor = device or (
            os.major(metadata.st_dev),
            os.minor(metadata.st_dev),
        )
        mapped_inode = metadata.st_ino if inode is None else inode
        pathname = str(path) if mapped_path is None else mapped_path
        return (
            f"{start:x}-{end:x} {permissions} 00000000 "
            f"{device_major:02x}:{device_minor:02x} {mapped_inode} {pathname}\n"
        )

    def _sha256(self, path: Path | None = None) -> str:
        return hashlib.sha256((path or self.library_path).read_bytes()).hexdigest()

    def _capture(
        self,
        *,
        path: Path | None = None,
        entrypoints: dict[str, int] | None = None,
        expected_sha256: str | None = None,
        snapshots: tuple[str, str] | None = None,
    ) -> dict:
        maps = snapshots or (self.maps_text, self.maps_text)
        with mock.patch.object(mapped_images, "_read_maps", side_effect=maps):
            return mapped_images.capture(
                path or self.library_path,
                self.entrypoints if entrypoints is None else entrypoints,
                self._sha256() if expected_sha256 is None else expected_sha256,
            )

    def test_capture_pins_selected_elf_and_binds_both_symbols_to_x_mapping(
        self,
    ) -> None:
        """Match both functions without admitting wider scientific claims."""
        result = self._capture()
        selected = result["selected_library"]
        metadata = self.library_path.stat()
        self.assertEqual(
            set(selected),
            {
                "resolved_path",
                "size_bytes",
                "sha256",
                "device_major",
                "device_minor",
                "inode",
                "mtime_ns",
                "ctime_ns",
            },
        )
        self.assertEqual(selected["resolved_path"], str(self.library_path.resolve()))
        self.assertEqual(selected["size_bytes"], metadata.st_size)
        self.assertEqual(selected["sha256"], self._sha256())
        self.assertEqual(selected["device_major"], os.major(metadata.st_dev))
        self.assertEqual(selected["device_minor"], os.minor(metadata.st_dev))
        self.assertEqual(selected["inode"], metadata.st_ino)
        self.assertEqual(selected["mtime_ns"], metadata.st_mtime_ns)
        self.assertEqual(selected["ctime_ns"], metadata.st_ctime_ns)
        self.assertEqual(result["entrypoints"], self.entrypoints)
        self.assertEqual(len(result["images"]), 1)
        self.assertTrue(result["images"][0]["is_elf"])
        self.assertEqual(result["maps_pins"][0]["path"], str(self.library_path))
        self.assertIn("endpoint-only", result["scope"])
        self.assertEqual(
            result["complete_runtime_dependency_closure"], "NOT_ESTABLISHED"
        )
        self.assertEqual(result["transient_images"], "NOT_OBSERVED")
        self.assertEqual(result["resident_memory_bytes"], "NOT_ATTESTED")
        self.assertFalse(result["performance_admission"])
        self.assertNotIn("atime", selected)
        self.assertNotIn("timestamp", selected)

    def test_selected_path_symlink_resolves_to_mapped_target(self) -> None:
        """Bind a selected symlink to its actual backing inode."""
        symlink_path = self.root / "selected-link.so"
        symlink_path.symlink_to(self.library_path)
        result = self._capture(path=symlink_path)
        self.assertEqual(
            result["selected_library"]["resolved_path"],
            str(self.library_path.resolve()),
        )

    def test_inventories_multiple_elf_and_non_elf_file_mappings(self) -> None:
        """Do not pretend every file-backed mapping represents an ELF object."""
        second_elf = self.root / "another.so"
        data_file = self.root / "mapped data.bin"
        second_elf.write_bytes(b"\x7fELF" + b"second")
        data_file.write_bytes(b"not an ELF image")
        maps = (
            self.maps_text
            + self._map_row(second_elf, start=0x3000, end=0x4000, permissions="r--p")
            + self._map_row(data_file, start=0x5000, end=0x6000, permissions="rw-p")
        )
        result = self._capture(snapshots=(maps, maps))
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(
            [image["is_elf"] for image in result["images"]], [True, True, False]
        )
        self.assertEqual(
            [image["path"] for image in result["images"]],
            [str(self.library_path), str(second_elf), str(data_file)],
        )

    def test_counts_anonymous_and_pseudopath_mappings_at_each_endpoint(self) -> None:
        """Unpinned mappings remain limitations rather than dependency proof."""
        before = self.maps_text + "3000-4000 rw-p 0 00:00 0 [heap]\n"
        after = before + "5000-6000 rw-p 0 00:00 0\n"
        result = self._capture(snapshots=(before, after))
        observation = result["maps_observation"]
        self.assertEqual(observation["anonymous_mapping_count_before"], 0)
        self.assertEqual(observation["anonymous_mapping_count_after"], 1)
        self.assertEqual(observation["pseudopath_mapping_count_before"], 1)
        self.assertEqual(observation["pseudopath_mapping_count_after"], 1)
        self.assertIn("counted only", observation["non_file_backed_limitations"])

    def test_rejects_expected_digest_mismatch(self) -> None:
        """An observed inode does not replace the external retained-byte pin."""
        with self.assertRaisesRegex(mapped_images.MappingError, "SHA-256"):
            self._capture(expected_sha256="0" * 64)

    def test_rejects_non_elf_selected_library(self) -> None:
        """Ordinary data cannot impersonate selected native code."""
        self.library_path.write_bytes(b"not ELF")
        self.maps_text = self._map_row(self.library_path)
        with self.assertRaisesRegex(mapped_images.MappingError, "ELF"):
            self._capture()

    def test_rejects_same_bytes_mapped_from_another_inode(self) -> None:
        """Equal bytes elsewhere are not the actually mapped selected inode."""
        other_inode = self.root / "same bytes other inode.so"
        other_inode.write_bytes(self.library_path.read_bytes())
        other_map = self._map_row(other_inode)
        with self.assertRaisesRegex(mapped_images.MappingError, "inode"):
            self._capture(snapshots=(other_map, other_map))

    def test_rejects_mapping_device_or_inode_mismatch(self) -> None:
        """The pathname alone cannot certify a backing image."""
        metadata = self.library_path.stat()
        wrong_inode_map = self._map_row(self.library_path, inode=metadata.st_ino + 1)
        with self.assertRaisesRegex(mapped_images.MappingError, "inode"):
            self._capture(snapshots=(wrong_inode_map, wrong_inode_map))

    def test_rejects_entrypoint_device_inode_mismatch(self) -> None:
        """Public functions must belong to the selected backing file."""
        metadata = self.library_path.stat()
        wrong_device_map = self._map_row(
            self.library_path,
            device=(os.major(metadata.st_dev) + 1, os.minor(metadata.st_dev)),
        )
        with self.assertRaisesRegex(mapped_images.MappingError, "inode"):
            self._capture(snapshots=(wrong_device_map, wrong_device_map))

    def test_rejects_missing_empty_and_boolean_entrypoint_addresses(self) -> None:
        """Every owner requires a complete pair of genuine positive addresses."""
        invalid_entrypoints = (
            {},
            {"owner-0:xtbloom_compute": 0x1010},
            {
                "owner-0:xtbloom_compute": True,
                "owner-0:xtbloom_context_create": 0x1020,
            },
            {
                "owner-0:xtbloom_compute": 0,
                "owner-0:xtbloom_context_create": 0x1020,
            },
        )
        for entrypoints in invalid_entrypoints:
            with (
                self.subTest(entrypoints=entrypoints),
                self.assertRaises(mapped_images.MappingError),
            ):
                self._capture(entrypoints=entrypoints)

    def test_rejects_entrypoint_outside_executable_mapping(self) -> None:
        """Data segments and adjacent addresses are not valid code locations."""
        non_executable = self._map_row(self.library_path, permissions="r--p")
        with self.assertRaisesRegex(mapped_images.MappingError, "executable"):
            self._capture(snapshots=(non_executable, non_executable))

        outside = dict(self.entrypoints)
        outside["owner-0:xtbloom_context_create"] = 0x2010
        with self.assertRaisesRegex(mapped_images.MappingError, "executable"):
            self._capture(entrypoints=outside)

    def test_rejects_unstable_file_backed_maps_entries(self) -> None:
        """Do not combine changing mappings into a purported stable endpoint."""
        changed = self._map_row(self.library_path, start=0x3000, end=0x4000)
        with self.assertRaisesRegex(mapped_images.MappingError, "changed"):
            self._capture(snapshots=(self.maps_text, changed))

    def test_detects_bytes_changed_while_hashing(self) -> None:
        """Byte rereads catch writes within one filesystem timestamp quantum."""
        real_read = os.read
        changed = False

        def mutate_during_read(descriptor: int, count: int) -> bytes:
            nonlocal changed
            block = real_read(descriptor, count)
            if block and not changed:
                changed = True
                contents = self.library_path.read_bytes()
                self.library_path.write_bytes(contents[:-1] + b"X")
            return block

        with (
            mock.patch.object(
                mapped_images, "_read_maps", side_effect=(self.maps_text,) * 2
            ),
            mock.patch.object(mapped_images.os, "read", side_effect=mutate_during_read),
            self.assertRaisesRegex(mapped_images.MappingError, "changed while hashing"),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, self._sha256())

    def test_byte_change_is_detected_when_inode_timestamps_do_not_change(self) -> None:
        """Metadata equality cannot substitute for stable retained bytes."""
        frozen = self.library_path.stat()
        expected = self._sha256()
        real_read = os.read
        changed = False

        def mutate_during_read(descriptor: int, count: int) -> bytes:
            nonlocal changed
            block = real_read(descriptor, count)
            if block and not changed:
                changed = True
                self.library_path.write_bytes(
                    self.library_path.read_bytes()[:-1] + b"X"
                )
            return block

        with (
            mock.patch.object(
                mapped_images, "_read_maps", side_effect=(self.maps_text,) * 2
            ),
            mock.patch.object(mapped_images.os, "stat", return_value=frozen),
            mock.patch.object(mapped_images.os, "fstat", return_value=frozen),
            mock.patch.object(mapped_images.os, "read", side_effect=mutate_during_read),
            self.assertRaisesRegex(mapped_images.MappingError, "bytes changed"),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, expected)

    def test_growing_file_does_not_extend_the_hash_extent(self) -> None:
        """Reject growth after the initial extent instead of following a writer."""
        expected = self._sha256()
        real_read = os.read
        read_calls = []

        def append_during_read(descriptor: int, count: int) -> bytes:
            read_calls.append(count)
            block = real_read(descriptor, count)
            if block:
                with self.library_path.open("ab") as output:
                    output.write(b"X")
            return block

        with (
            mock.patch.object(
                mapped_images, "_read_maps", side_effect=(self.maps_text,) * 2
            ),
            mock.patch.object(mapped_images.os, "read", side_effect=append_during_read),
            self.assertRaisesRegex(mapped_images.MappingError, "length changed"),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, expected)
        self.assertEqual(len(read_calls), 2)

    def test_detects_path_replaced_while_hashing(self) -> None:
        """Replaced paths cannot borrow an old open descriptor's identity."""
        replacement = self.root / "replacement.so"
        replacement.write_bytes(b"\x7fELF" + b"replacement")
        real_read = os.read
        replaced = False

        def replace_during_read(descriptor: int, count: int) -> bytes:
            nonlocal replaced
            block = real_read(descriptor, count)
            if block and not replaced:
                replaced = True
                os.replace(replacement, self.library_path)
            return block

        with (
            mock.patch.object(
                mapped_images, "_read_maps", side_effect=(self.maps_text,) * 2
            ),
            mock.patch.object(
                mapped_images.os, "read", side_effect=replace_during_read
            ),
            self.assertRaises(mapped_images.MappingError),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, self._sha256())

    def test_detects_path_replaced_before_second_maps_snapshot(self) -> None:
        """Retain the final pathname check after reading mappings again."""
        replacement = self.root / "replacement.so"
        replacement.write_bytes(b"\x7fELF" + b"replacement")
        calls = 0

        def maps_with_replacement() -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                os.replace(replacement, self.library_path)
            return self.maps_text

        with (
            mock.patch.object(
                mapped_images, "_read_maps", side_effect=maps_with_replacement
            ),
            self.assertRaises(mapped_images.MappingError),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, self._sha256())

    def test_rejects_invalid_platform_and_maps_read_errors(self) -> None:
        """Unsupported observations fail without an alternative probe."""
        with mock.patch.object(mapped_images.sys, "platform", "darwin"):
            with self.assertRaisesRegex(mapped_images.MappingError, "Linux"):
                mapped_images._read_maps()
            with self.assertRaisesRegex(mapped_images.MappingError, "Linux"):
                mapped_images.capture(
                    self.library_path, self.entrypoints, self._sha256()
                )

        with (
            mock.patch.object(
                mapped_images,
                "_read_maps",
                side_effect=mapped_images.MappingError("read failed"),
            ),
            self.assertRaisesRegex(mapped_images.MappingError, "read failed"),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, self._sha256())

    def test_read_maps_reads_to_eof_and_rejects_oversize(self) -> None:
        """Bound total bytes while distinguishing a short read from EOF."""
        with (
            mock.patch.object(mapped_images.os, "open", return_value=31),
            mock.patch.object(
                mapped_images.os, "read", side_effect=[b"one complete line\n", b""]
            ) as read,
            mock.patch.object(mapped_images.os, "close") as close,
        ):
            self.assertEqual(mapped_images._read_maps(), "one complete line\n")
        self.assertEqual(
            read.call_args_list,
            [mock.call(31, 64 * 1024), mock.call(31, 64 * 1024)],
        )
        close.assert_called_once_with(31)

        with (
            mock.patch.object(mapped_images, "_MAX_MAPS_BYTES", 4),
            mock.patch.object(mapped_images.os, "open", return_value=32),
            mock.patch.object(mapped_images.os, "read", return_value=b"12345\n"),
            mock.patch.object(mapped_images.os, "close"),
            self.assertRaisesRegex(mapped_images.MappingError, "size limit"),
        ):
            mapped_images._read_maps()

    def test_read_maps_consumes_newline_ended_prefix_and_rejects_bad_suffix(
        self,
    ) -> None:
        """Do not omit a deleted mapping after a complete short-read boundary."""
        prefix = b"1000-2000 rw-p 00000000 00:00 0 [heap]\n"
        suffix = b"3000-4000 r--p 00000000 08:01 1 /tmp/removed (deleted)\n"
        with (
            mock.patch.object(mapped_images.os, "open", return_value=34),
            mock.patch.object(
                mapped_images.os, "read", side_effect=[prefix, suffix, b""]
            ),
            mock.patch.object(mapped_images.os, "close") as close,
        ):
            snapshot = mapped_images._read_maps()
        self.assertEqual(snapshot, (prefix + suffix).decode())
        with self.assertRaisesRegex(mapped_images.MappingError, "deleted"):
            mapped_images.parse_maps(snapshot)
        close.assert_called_once_with(34)

    def test_read_maps_closes_descriptor_after_eof_probe_error(self) -> None:
        """An unsuccessful EOF check never admits the already-read prefix."""
        with (
            mock.patch.object(mapped_images.os, "open", return_value=35),
            mock.patch.object(
                mapped_images.os,
                "read",
                side_effect=[b"complete line\n", OSError("EOF probe failed")],
            ),
            mock.patch.object(mapped_images.os, "close") as close,
            self.assertRaisesRegex(mapped_images.MappingError, "EOF probe failed"),
        ):
            mapped_images._read_maps()
        close.assert_called_once_with(35)

    def test_read_maps_preserves_records_split_across_kernel_reads(self) -> None:
        """A short read within a record is valid when the bounded suffix arrives."""
        with (
            mock.patch.object(mapped_images.os, "open", return_value=36),
            mock.patch.object(
                mapped_images.os,
                "read",
                side_effect=[b"1000-2000 rw-p", b" 00000000 00:00 0 [heap]\n", b""],
            ),
            mock.patch.object(mapped_images.os, "close") as close,
        ):
            records = mapped_images.parse_maps(mapped_images._read_maps())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["path"], "[heap]")
        close.assert_called_once_with(36)

    def test_read_maps_bounds_nonterminating_short_reads(self) -> None:
        """An input that never reaches EOF still fails at the total-byte cap."""
        with (
            mock.patch.object(mapped_images, "_MAX_MAPS_BYTES", 4),
            mock.patch.object(mapped_images.os, "open", return_value=37),
            mock.patch.object(mapped_images.os, "read", return_value=b"x") as read,
            mock.patch.object(mapped_images.os, "close") as close,
            self.assertRaisesRegex(mapped_images.MappingError, "size limit"),
        ):
            mapped_images._read_maps()
        self.assertEqual(read.call_count, 5)
        self.assertEqual(read.call_args_list[-1], mock.call(37, 1))
        close.assert_called_once_with(37)

    def test_read_maps_rejects_partial_final_line(self) -> None:
        """A snapshot ending within a kernel record is incomplete."""
        with (
            mock.patch.object(mapped_images.os, "open", return_value=33),
            mock.patch.object(mapped_images.os, "read", side_effect=[b"partial", b""]),
            mock.patch.object(mapped_images.os, "close"),
            self.assertRaisesRegex(mapped_images.MappingError, "truncated"),
        ):
            mapped_images._read_maps()

    def test_read_maps_closes_descriptor_after_read_error(self) -> None:
        """Failed kernel reads release their process-map descriptor."""
        with (
            mock.patch.object(mapped_images.os, "open", return_value=41),
            mock.patch.object(
                mapped_images.os, "read", side_effect=OSError("read error")
            ),
            mock.patch.object(mapped_images.os, "close") as close,
            self.assertRaisesRegex(mapped_images.MappingError, "read error"),
        ):
            mapped_images._read_maps()
        close.assert_called_once_with(41)

    def test_capture_closes_file_descriptor_after_hash_read_error(self) -> None:
        """Hash errors release descriptors owned by the observer."""
        opened: list[int] = []
        closed: list[int] = []
        real_open = os.open
        real_close = os.close

        def tracked_open(path: str, flags: int) -> int:
            descriptor = real_open(path, flags)
            opened.append(descriptor)
            return descriptor

        def tracked_close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        with (
            mock.patch.object(mapped_images, "_read_maps", return_value=self.maps_text),
            mock.patch.object(mapped_images.os, "open", side_effect=tracked_open),
            mock.patch.object(
                mapped_images.os, "read", side_effect=OSError("hash read error")
            ),
            mock.patch.object(mapped_images.os, "close", side_effect=tracked_close),
            self.assertRaisesRegex(mapped_images.MappingError, "hash read error"),
        ):
            mapped_images.capture(self.library_path, self.entrypoints, self._sha256())
        self.assertTrue(opened)
        self.assertCountEqual(opened, closed)

    def test_library_entrypoints_extracts_addresses_without_calling_functions(
        self,
    ) -> None:
        """Inspect function addresses without executing native entrypoints."""
        function_type = ctypes.CFUNCTYPE(None)
        libraries = [ctypes.CDLL(None), ctypes.CDLL(None)]
        assigned_addresses = ((0x1010, 0x1020), (0x2010, 0x2020))
        for library, addresses in zip(libraries, assigned_addresses, strict=True):
            library.xtbloom_compute = ctypes.cast(
                ctypes.c_void_p(addresses[0]), function_type
            )
            library.xtbloom_context_create = ctypes.cast(
                ctypes.c_void_p(addresses[1]), function_type
            )

        self.assertEqual(
            mapped_images.library_entrypoints(libraries),
            {
                "owner-0:xtbloom_compute": 0x1010,
                "owner-0:xtbloom_context_create": 0x1020,
                "owner-1:xtbloom_compute": 0x2010,
                "owner-1:xtbloom_context_create": 0x2020,
            },
        )

    def test_library_entrypoints_rejects_non_cdll_and_missing_symbol(self) -> None:
        """A guessed pathname is not a complete live library handle."""
        with self.assertRaisesRegex(mapped_images.MappingError, "ctypes.CDLL"):
            mapped_images.library_entrypoints([object()])  # type: ignore[list-item]

        library = ctypes.CDLL(None)
        with self.assertRaisesRegex(mapped_images.MappingError, "xtbloom_compute"):
            mapped_images.library_entrypoints([library])


if __name__ == "__main__":
    unittest.main()
