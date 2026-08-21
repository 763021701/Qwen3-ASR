import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.prepare_real_raw_denoised_sft import (
    load_additional_synthetic_rows,
    load_rows,
    parse_additional_synthetic_specs,
    prepare,
    validate_pair_alignment,
)

_CSV_HEADER = ["source", "index", "audio_path", "text", "original", "duration"]


class PrepareRealRawDenoisedTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_pair_alignment_and_noise_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_path = root / "metadata_raw.jsonl"
            denoised_path = root / "metadata_denoised.jsonl"
            rows = [
                {
                    "audio_path": str(root / "from_lid_raw" / "case1_seg_0001.wav"),
                    "text": "language English<asr_text>specimen labelled left",
                    "duration": 1.0,
                },
                {
                    "audio_path": str(root / "from_lid_raw" / "case2_seg_0001.wav"),
                    "text": "second specimen",
                    "duration": 2.0,
                },
            ]
            denoised_rows = [
                {
                    **row,
                    "audio_path": row["audio_path"].replace("from_lid_raw", "from_lid_denoised"),
                }
                for row in rows
            ]
            self._write_jsonl(raw_path, rows)
            self._write_jsonl(denoised_path, denoised_rows)

            raw = load_rows(raw_path, sampling_source="real_raw", noise_aug=0, check_audio=False)
            denoised = load_rows(
                denoised_path,
                sampling_source="real_denoised",
                noise_aug=1,
                check_audio=False,
            )
            validate_pair_alignment(raw, denoised)

            self.assertEqual({row["noise_aug"] for row in raw}, {0})
            self.assertEqual({row["noise_aug"] for row in denoised}, {1})
            self.assertTrue(all(row["text"].startswith("language None<asr_text>") for row in raw))

    def test_prepare_keeps_raw_and_denoised_in_same_group_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_path = root / "metadata_raw.jsonl"
            denoised_path = root / "metadata_denoised.jsonl"
            test_path = root / "test.jsonl"
            raw_rows = [
                {"audio_path": str(root / "from_raw" / "case1_seg_0001.wav"), "text": "one", "duration": 1.0},
                {"audio_path": str(root / "from_raw" / "case2_seg_0001.wav"), "text": "two", "duration": 1.0},
            ]
            denoised_rows = [
                {
                    **row,
                    "audio_path": row["audio_path"].replace("/from_raw/", "/from_denoised/"),
                }
                for row in raw_rows
            ]
            self._write_jsonl(raw_path, raw_rows)
            self._write_jsonl(denoised_path, denoised_rows)
            self._write_jsonl(
                test_path,
                [{"audio_path": str(root / "test.wav"), "text": "test sample", "duration": 1.0}],
            )

            output_dir = root / "prepared"
            report = prepare(
                SimpleNamespace(
                    raw_jsonl=str(raw_path),
                    denoised_jsonl=str(denoised_path),
                    test_source=str(test_path),
                    output_dir=str(output_dir),
                    train_jsonl="",
                    dev_jsonl="",
                    test_jsonl="",
                    dev_fraction=0.5,
                    seed=42,
                    check_audio=0,
                )
            )

            train = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(line) for line in (output_dir / "dev.jsonl").read_text().splitlines()]
            self.assertEqual(report["pair_alignment"]["paired_segments"], 2)
            self.assertEqual({row["source_group"] for row in train}.isdisjoint(
                {row["source_group"] for row in dev}
            ), True)
            self.assertEqual({row["noise_aug"] for row in train}, {0, 1})
            self.assertEqual({row["aug"] for row in dev}, {0})


class AdditionalSyntheticTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_CSV_HEADER)
            writer.writerows(rows)

    @staticmethod
    def _touch(paths: list[Path]) -> None:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @staticmethod
    def _spec_csv_rows(root: Path, texts_by_voice: dict[str, list[tuple[str, str, str]]]) -> list[list[str]]:
        """Build CSV rows: voice -> list of (index, text, original)."""
        rows = []
        for voice in sorted(texts_by_voice):
            for index, text, original in texts_by_voice[voice]:
                rows.append(
                    [voice, index, str(root / voice / f"{index}.wav"), text, original, "1.0"]
                )
        return rows

    def test_parse_specs(self):
        specs = parse_additional_synthetic_specs(["a=x.csv", "b=y.csv"])
        self.assertEqual(specs, [("a", Path("x.csv")), ("b", Path("y.csv"))])
        with self.assertRaises(ValueError):
            parse_additional_synthetic_specs(["no-equals"])
        with self.assertRaises(ValueError):
            parse_additional_synthetic_specs(["a=x.csv", "a=y.csv"])

    def test_voice_sampling_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "meta.csv"
            rows = self._spec_csv_rows(
                root,
                {
                    "VA": [("1", "one", "1"), ("2", "two", "2")],
                    "VB": [("1", "one", "1"), ("2", "two", "2")],
                    "VC": [("1", "one", "1"), ("2", "two", "2")],
                },
            )
            self._write_csv(csv_path, rows)

            def run() -> list[dict]:
                kept, report = load_additional_synthetic_rows(
                    "spec",
                    csv_path,
                    max_per_text=2,
                    check_audio=False,
                    seen_audio=set(),
                )
                return kept, report

            kept_first, report_first = run()
            kept_second, report_second = run()
            self.assertEqual(kept_first, kept_second)
            self.assertEqual(report_first, report_second)

            self.assertEqual(len(kept_first), 4)
            self.assertEqual(report_first["normalized_text_groups"], 2)
            self.assertEqual(report_first["excluded_rows"], 2)
            self.assertEqual(report_first["voice_counts"], {"VA": 2, "VB": 1, "VC": 1})

            by_text = {}
            for row in kept_first:
                by_text.setdefault(row["text"].split("<asr_text>")[1], []).append(row)
            for label, group in by_text.items():
                voices = {row["segment_id"].split("/")[1].rsplit("_", 1)[0] for row in group}
                self.assertEqual(len(voices), len(group), f"voices not distinct for {label!r}")

    def test_grouping_uses_normalized_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "meta.csv"
            rows = self._spec_csv_rows(
                root,
                {
                    "VA": [("1", "1 cm", "1 cm")],
                    "VB": [("1", "one cm", "1 cm")],
                    "VC": [("1", "1cm", "1 cm")],
                },
            )
            self._write_csv(csv_path, rows)
            kept, report = load_additional_synthetic_rows(
                "spec", csv_path, max_per_text=2, check_audio=False, seen_audio=set()
            )
            self.assertEqual(report["normalized_text_groups"], 1)
            self.assertEqual(len(kept), 2)
            self.assertEqual(report["voice_counts"], {"VA": 1, "VB": 1})

    def test_labels_and_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "meta.csv"
            rows = self._spec_csv_rows(root, {"VA": [("0001", "一 c m", "1 cm")]})
            self._write_csv(csv_path, rows)
            kept, _ = load_additional_synthetic_rows(
                "cantonese_measure", csv_path, max_per_text=2, check_audio=False, seen_audio=set()
            )
            row = kept[0]
            self.assertEqual(row["text"], "language None<asr_text>1 cm")
            self.assertEqual(row["aug"], 1)
            self.assertEqual(row["noise_aug"], 1)
            self.assertEqual(row["sampling_source"], "cantonese_measure")
            self.assertEqual(row["segment_id"], "cantonese_measure/VA_0001")
            self.assertEqual(row["duration_sec"], 1.0)

    def test_field_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = self._spec_csv_rows(root, {"VA": [("1", "one", "1")]})

            missing_column = root / "missing.csv"
            with missing_column.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source", "index", "audio_path", "text", "duration"])
                writer.writerow(base[0][:4] + base[0][5:])
            with self.assertRaises(ValueError):
                load_additional_synthetic_rows(
                    "s", missing_column, max_per_text=2, check_audio=False, seen_audio=set()
                )

            empty_field = root / "empty.csv"
            self._write_csv(empty_field, [[r[0], r[1], r[2], "", r[4], r[5]] for r in base])
            with self.assertRaises(ValueError):
                load_additional_synthetic_rows(
                    "s", empty_field, max_per_text=2, check_audio=False, seen_audio=set()
                )

            bad_duration = root / "bad_duration.csv"
            self._write_csv(bad_duration, [[r[0], r[1], r[2], r[3], r[4], "abc"] for r in base])
            with self.assertRaises(ValueError):
                load_additional_synthetic_rows(
                    "s", bad_duration, max_per_text=2, check_audio=False, seen_audio=set()
                )

            duplicate_audio = root / "duplicate.csv"
            self._write_csv(duplicate_audio, base + [list(base[0])])
            with self.assertRaises(ValueError):
                load_additional_synthetic_rows(
                    "s", duplicate_audio, max_per_text=2, check_audio=False, seen_audio=set()
                )

            missing_audio = root / "missing_audio.csv"
            self._write_csv(missing_audio, base)
            with self.assertRaises(FileNotFoundError):
                load_additional_synthetic_rows(
                    "s", missing_audio, max_per_text=2, check_audio=True, seen_audio=set()
                )

    def test_prepare_with_additional_csvs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_path = root / "metadata_raw.jsonl"
            denoised_path = root / "metadata_denoised.jsonl"
            test_path = root / "test.jsonl"
            real = [
                {"audio_path": str(root / "from_raw" / "case1_seg_0001.wav"), "text": "one", "duration": 1.0},
                {"audio_path": str(root / "from_raw" / "case2_seg_0001.wav"), "text": "two", "duration": 1.0},
            ]
            denoised = [
                {**row, "audio_path": row["audio_path"].replace("/from_raw/", "/from_denoised/")}
                for row in real
            ]
            self._write_jsonl(raw_path, real)
            self._write_jsonl(denoised_path, denoised)
            self._write_jsonl(test_path, [{"audio_path": str(root / "test.wav"), "text": "test", "duration": 1.0}])
            self._touch(
                [Path(row["audio_path"]) for row in real + denoised]
                + [root / "test.wav"]
                + [root / voice / f"{i}.wav" for voice in ("VA", "VB", "VC") for i in ("1", "2")]
            )

            csv_a = root / "specimen.csv"
            csv_b = root / "measure.csv"
            self._write_csv(
                csv_a,
                self._spec_csv_rows(
                    root,
                    {
                        "VA": [("1", "one", "1"), ("2", "two", "2")],
                        "VB": [("1", "one", "1"), ("2", "two", "2")],
                        "VC": [("1", "one", "1"), ("2", "two", "2")],
                    },
                ),
            )
            # second source reuses a different audio dir but shares voice names
            self._write_csv(
                csv_b,
                [
                    [r[0], r[1], r[2].replace("VA/", "MA_").replace("VB/", "MB_").replace("VC/", "MC_"), r[3], r[4], r[5]]
                    for r in self._spec_csv_rows(
                        root,
                        {
                            "VA": [("1", "a b", "a b")],
                            "VB": [("1", "c d", "c d")],
                        },
                    )
                ],
            )
            self._touch([root / "MA_1.wav", root / "MB_1.wav"])

            output_dir = root / "prepared"
            report = prepare(
                SimpleNamespace(
                    raw_jsonl=str(raw_path),
                    denoised_jsonl=str(denoised_path),
                    test_source=str(test_path),
                    output_dir=str(output_dir),
                    train_jsonl="",
                    dev_jsonl="",
                    test_jsonl="",
                    dev_fraction=0.2,
                    use_test_as_dev=1,
                    synthetic_jsonl="",
                    synthetic_metadata_csv="",
                    synthetic_audio_root="",
                    synthetic_exclude_priorities="test_only",
                    additional_synthetic_csv=[f"spec={csv_a}", f"meas={csv_b}"],
                    additional_synthetic_max_per_text=2,
                    seed=42,
                    check_audio=1,
                )
            )

            train = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(train), 4 + 4 + 2)
            self.assertEqual(report["additional_synthetic"]["spec"]["kept"]["rows"], 4)
            self.assertEqual(report["additional_synthetic"]["meas"]["kept"]["rows"], 2)
            self.assertEqual(
                report["additional_synthetic"]["spec"]["voice_counts"],
                {"VA": 2, "VB": 1, "VC": 1},
            )

            audios = [row["audio"] for row in train]
            self.assertEqual(len(audios), len(set(audios)))
            synthetic_ids = [
                row["segment_id"] for row in train if row["sampling_source"] in {"spec", "meas"}
            ]
            self.assertEqual(len(synthetic_ids), len(set(synthetic_ids)))
            self.assertEqual(
                {row["sampling_source"] for row in train},
                {"real_raw", "real_denoised", "spec", "meas"},
            )
            for row in train:
                if row["sampling_source"] in {"spec", "meas"}:
                    self.assertEqual(row["aug"], 1)
                    self.assertEqual(row["noise_aug"], 1)
                    self.assertTrue(row["text"].startswith("language None<asr_text>"))


if __name__ == "__main__":
    unittest.main()
