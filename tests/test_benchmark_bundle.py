import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from maple_guard.benchmarks.benchmark_bundle import load_bundle, appworld_cases, normalized_rows
from maple_guard import run_appworld

class BundleTests(unittest.TestCase):
    def write(self, folder, benchmark="appworld", rows=None, **extra):
        if rows is None:
            rows=[{"benchmark":benchmark,"task_id":benchmark+":a_1","native_task_id":"a_1",
                   "split":"test_normal","data":{"instruction":"Read my calendar","datetime":"2023-01-01"}}]
        p=Path(folder)/"bundle.json"
        p.write_text(json.dumps(dict(benchmark=benchmark,task_count=len(rows),tasks=rows,**extra)))
        return p
    def test_bundle_manifest_hash_and_identity_preserved(self):
        with tempfile.TemporaryDirectory() as f:
            p=self.write(f)
            bundle, manifest=load_bundle(p,"appworld")
            cases=appworld_cases(bundle,str(p))
            self.assertEqual(cases[0].task_id,"a_1")
            self.assertEqual(cases[0].instruction,"Read my calendar")
            self.assertEqual(manifest["ordered_task_ids"],["appworld:a_1"])
            self.assertEqual(len(manifest["source_sha256"]),64)
    def test_invalid_counts_duplicates_and_wrong_benchmark_fail(self):
        with tempfile.TemporaryDirectory() as f:
            p=self.write(f); obj=json.loads(p.read_text())
            for change, expected in [
                (dict(task_count=2),"count"),
                (dict(tasks=obj["tasks"]*2,task_count=2),"Duplicate"),
                (dict(benchmark="csqa"),"benchmark")]:
                p.write_text(json.dumps(dict(obj,**change)))
                with self.assertRaisesRegex(ValueError,expected): load_bundle(p,"appworld")
    def test_empty_instruction_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as f:
            p=self.write(f); obj=json.loads(p.read_text())
            obj["tasks"][0]["data"]["instruction"]=""
            with self.assertRaisesRegex(ValueError,"instruction"): appworld_cases(obj,str(p))
    def test_csqa_nested_question_is_unwrapped_not_stringified(self):
        with tempfile.TemporaryDirectory() as f:
            rows=[{"benchmark":"csqa","task_id":"csqa:x","split":"dev","data":{
                "id":"x","question":{"stem":"Where?","choices":[{"label":"A","text":"home"},{"label":"B","text":"school"}]},
                "answerKey":"B"}}]
            obj,_=load_bundle(self.write(f,"csqa",rows))
            row=normalized_rows(obj)[0]
            self.assertEqual((row["question"],row["id"],row["answerKey"]),("Where?","csqa:x","B"))
            self.assertEqual(len(row["choices"]),2)
    def test_bundle_path_never_rebuilds_entire_native_index(self):
        with tempfile.TemporaryDirectory() as f:
            p=self.write(f)
            args=SimpleNamespace(benchmark_bundle=str(p),dataset="",rebuild_index=False,tasks=200,seed=42)
            with patch.object(run_appworld,"write_index",side_effect=AssertionError("wrong dataset")):
                manifest=run_appworld.ensure_appworld_index(args)
                cases=run_appworld.select_cases(args)
            self.assertEqual(len(cases),1)
            self.assertEqual(manifest["evaluation_protocol"],"appworld_action_selection_proxy")
            self.assertFalse(manifest["native_execution"])
    def test_bundle_and_dataset_are_mutually_exclusive(self):
        args=SimpleNamespace(benchmark_bundle="bundle.json",dataset="another.jsonl",rebuild_index=False)
        with self.assertRaisesRegex(ValueError,"dataset"): run_appworld.ensure_appworld_index(args)
    def test_real_commonsenseqa_name_supported(self):
        with tempfile.TemporaryDirectory() as f:
            rows=[{"benchmark":"commonsenseqa","task_id":"csqa:x","data":{"question":{"stem":"Why?", "choices":[{"label":"A","text":"yes"}]},"answerKey":"A"}}]
            bundle,manifest=load_bundle(self.write(f,"commonsenseqa",rows),"csqa")
            self.assertEqual(manifest["benchmark"],"csqa")
            self.assertEqual(normalized_rows(bundle)[0]["question"],"Why?")
    def test_row_order_preserved(self):
        with tempfile.TemporaryDirectory() as f:
            rows=[{"benchmark":"mmlu","task_id":i,"subject":"anatomy","data":{"question":i,"choices":["a","b"],"answer":"B"}} for i in ["z","a"]]
            obj,_=load_bundle(self.write(f,"mmlu",rows))
            self.assertEqual([r["id"] for r in normalized_rows(obj)],["z","a"])

if __name__=="__main__": unittest.main()
