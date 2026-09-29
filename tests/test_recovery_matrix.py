import contextlib, io, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from tools import run_appworld_matrix as matrix

class RecoveryMatrixTests(unittest.TestCase):
    def parse(self,*extra):
        with patch("sys.argv",["matrix","--bundle","/tmp/tasks.json","--services","/tmp/services.json","--run-root","/tmp/new-run",*extra]):
            return matrix.arguments()
    def services(self):
        return {n:{"base_url":"http://"+n+"/v1","model":n} for n in ("inference1","embedding")}
    def test_timeout_reaches_worker_without_changing_budget(self):
        job=matrix.build_job(self.parse("--full-judge-timeout","600"),"amemguard_full",42,self.services())
        cmd=job["command"]
        self.assertEqual(cmd[cmd.index("--full-judge-timeout")+1],"600.0")
        self.assertNotIn("--full-judge-max-tokens",cmd)
        self.assertEqual(job["transport_configuration"]["judge_timeout_seconds"],600.0)
    def test_nonpositive_timeout_rejected(self):
        for x in ("0","-1"):
            with self.subTest(value=x),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.parse("--full-judge-timeout",x)
    def test_default_preserves_timeout(self):
        job=matrix.build_job(self.parse(),"amemguard_full",42,self.services())
        self.assertNotIn("--full-judge-timeout",job["command"])
    def test_context_flags_and_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"asset.json";p.write_text("{}")
            settings={"context-policy":"bounded_recent_v1","target-tokenizer":"/models/gemma","judge-tokenizer":"/models/qwen","target-context-limit":"32768","judge-context-limit":"32768","context-margin":"128","target-content-format":"openai","judge-content-format":"string"}
            flags=[v for k,x in settings.items() for v in ("--agentsafe-"+k,x)]
            args=self.parse("--agentsafe-profile","paper_v2_adapted","--agentsafe-policy",str(p),"--agentsafe-criteria",str(p),"--agentsafe-calibration-manifest",str(p),"--agentsafe-threshold","0.182565380021364",*flags)
            job=matrix.build_job(args,"agentsafe_full",42,self.services());cmd=job["command"]
            self.assertEqual(job["status"],"prepared")
            for k,v in settings.items():
                self.assertEqual(cmd[cmd.index("--agentsafe-"+k)+1],v)
            meta=job["baseline_configuration"]
            self.assertFalse(meta["official_configuration_recovered"])
            self.assertEqual(meta["context_policy"],"bounded_recent_v1")
            self.assertEqual(meta["target_context_limit"],32768)
            self.assertEqual(meta["judge_context_limit"],32768)
            self.assertEqual(meta["target_content_format"],"openai")
