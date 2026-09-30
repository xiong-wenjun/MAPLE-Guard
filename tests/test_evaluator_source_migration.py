"""The evaluator-only recovery authorizes exact source bytes, never other identity changes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_task_checkpoint as fixtures
from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
from evaluate.defense_methods.base import OfficialDefenseState


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class EvaluatorSourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TaskCheckpointTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.cp,self.args,self.bundle=self.fixture.ck,self.fixture.args,self.fixture.bundle
        self.args.method='gsafeguard';self.args.task_mode='qa';self.args.strict_comparison=True
        self.args.response_budget_policy='fail_task'
        Path(self.args.baseline_state_path).unlink()
        runtime=ComparisonRuntime(self.args)
        runtime.ledger={'memory-state':{'private':'unchanged'}}
        runtime.selected={0:[self.bundle.private_memories[0][0]]}
        runtime.consumed={0:{'m1':self.bundle.private_memories[0][0]}}
        runtime.received={0:[{'taints':['external']} ]}
        self.args._full_baseline_runtime=runtime
        self.fixture.state['official_defense_state']=OfficialDefenseState(guardian_history=[{0:'past safe output'}],kicked_agents={1})
        self.old=json.loads((Path(__file__).parent/'fixtures/evaluator_recovery_86a8b158_source.json').read_text())
        self.original_identity=self.cp.build_identity

    def save_old(self):
        identity=self.original_identity(self.args);identity['source']=self.old
        with patch.object(self.cp,'build_identity',return_value=identity):self.fixture.save()
        return identity

    def authorize(self,saved):
        current=self.original_identity(self.args)
        changes={key:{'before':saved['source'][key],'after':current['source'][key]}
                 for key in current['source'] if saved['source'].get(key)!=current['source'][key]}
        manifest={'schema':1,'purpose':'external_evaluator_pending_recovery','prompt_path_relocations':{},
                  'source_before_sha256':digest(saved['source']),'source_after_sha256':digest(current['source']),
                  'changes':changes}
        path=self.fixture.root/'operator-authorized-source.json'
        path.write_text(json.dumps(manifest,sort_keys=True)+'\n')
        self.args.checkpoint_evaluator_recovery_manifest=str(path)
        self.args.checkpoint_evaluator_recovery_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.args.resume_task_checkpoint=True
        return path

    def test_nonzero_old_full_state_restores_without_task_memory_or_runtime_changes(self):
        saved=self.save_old();self.authorize(saved)
        encoded=self.cp._encode(self.cp._capture_bundle(self.bundle))
        state=copy.deepcopy(self.fixture.state)
        try:data=self.fixture.load()
        except self.cp.CheckpointError as error:self.fail('Approved exact evaluator recovery rejected: '+str(error))
        fresh=self.fixture.Bundle(self.args,self.bundle.entry_cls)
        self.cp.restore_bundle(fresh,data)
        self.assertEqual(data['stream_state'],state)
        self.assertEqual(self.cp._encode(self.cp._capture_bundle(fresh)),encoded)
        self.assertIs(self.args._full_baseline_runtime.selected[0][0],fresh.private_memories[0][0])
        self.assertEqual(data['source_transition']['next_task_index'],1)
        self.assertIsNone(data['budget_transition'])
        self.assertEqual(len(self.args._checkpoint_source_history),1)
        audit=json.loads((Path(data['failed_attempt_path'])/'recovery.json').read_text())
        self.assertEqual(audit['source_transition'],data['source_transition'])
        self.fixture.bundle=fresh;self.fixture.save()
        self.assertIsNone(self.fixture.load()['source_transition'])
        self.assertEqual(len(self.args._checkpoint_source_history),1)
        self.assertTrue(self.cp.checkpoint_summary(self.args)['uniform_budget'])
        self.assertFalse(self.cp.checkpoint_summary(self.args)['uniform_source'])

    def test_strict_default_still_rejects_old_source_before_disk_mutation(self):
        self.save_old();before=Path(self.args.out).read_bytes()
        with self.assertRaisesRegex(self.cp.CheckpointError,'identity'):self.fixture.load()
        self.assertEqual(Path(self.args.out).read_bytes(),before)

    def test_authorization_does_not_allow_config_budget_prompt_environment_or_extra_source_changes(self):
        for alteration in ('seed','budget','prompt','environment','extra-source'):
            with self.subTest(alteration=alteration):
                saved=self.save_old();self.authorize(saved)
                current=self.original_identity(self.args)
                if alteration=='seed':self.args.seed=999
                elif alteration=='budget':self.args.chat_max_tokens=4096;self.args.checkpoint_allow_budget_change=True
                elif alteration=='prompt':current['prompts']['bundle']['sha256']='unapproved'
                elif alteration=='environment':current['environment']['CHAT_MAX_TOKENS']='4096'
                else:current['source']['maple_guard/run_mmlu.py']='unapproved'
                before=Path(self.args.out).read_bytes()
                with patch.object(self.cp,'build_identity',return_value=current) if alteration in ('prompt','environment','extra-source') else patch.object(self.cp,'_hash',wraps=self.cp._hash):
                    with self.assertRaises(self.cp.CheckpointError):self.fixture.load()
                self.assertEqual(Path(self.args.out).read_bytes(),before)
                if alteration=='seed':self.args.seed=4
                if alteration=='budget':del self.args.chat_max_tokens;self.args.checkpoint_allow_budget_change=False
                # Next independent rejection uses a fresh old generation.
                import shutil
                shutil.rmtree(self.args.task_checkpoint_dir)
                vars(self.args).pop('checkpoint_evaluator_recovery_manifest',None)
                vars(self.args).pop('checkpoint_evaluator_recovery_sha256',None)

    def test_wrong_manifest_hash_wrong_old_fixture_and_unrelated_approved_file_are_rejected(self):
        saved=self.save_old();path=self.authorize(saved)
        self.args.checkpoint_evaluator_recovery_sha256='0'*64
        with self.assertRaises(self.cp.CheckpointError):self.fixture.load()
        self.args.checkpoint_evaluator_recovery_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        spec=json.loads(path.read_text());spec['changes']['maple_guard/run_mmlu.py']={'before':'0'*64,'after':'1'*64}
        path.write_text(json.dumps(spec)+'\n');self.args.checkpoint_evaluator_recovery_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(self.cp.CheckpointError):self.fixture.load()
        self.assertEqual(digest(saved['source']),'eb0845601411ba607c57d55f9ee062885c0748a79017cf37722a948b736d0e81')

    def test_controller_rebases_only_authorized_source_and_keeps_original_working_identity(self):
        from tools import run_recovery_plan as recovery
        saved=self.save_old();authorization=self.authorize(saved)
        job={'run_id':'existing','directory':str(self.fixture.root),'method':'gsafeguard',
             'command':['python','-B','/old/snapshot/tools/run_instrumented.py','maple_guard.run_mmlu',
                        '--task-checkpoint-dir',self.args.task_checkpoint_dir,'--out',self.args.out,
                        '--chat-max-tokens','2048','--memory-run-id','unchanged','--seed','42']}
        with self.assertRaises(ValueError):recovery.resume_job(job)
        try:
            resumed=recovery.resume_job(job,evaluator_recovery={'manifest':str(authorization),'sha256':self.args.checkpoint_evaluator_recovery_sha256})
        except TypeError as error:self.fail('Source-rebased approved resume is missing: '+str(error))
        self.assertEqual(resumed['command'][2],str(recovery.ROOT/'tools/run_instrumented.py'))
        self.assertEqual(resumed['directory'],job['directory']);self.assertEqual(resumed['run_id'],job['run_id'])
        self.assertEqual(resumed['command'][4:len(job['command'])],job['command'][4:])
        self.assertIn('--resume-task-checkpoint',resumed['command'])
        self.assertIn('--checkpoint-evaluator-recovery-manifest',resumed['command'])
        self.assertNotIn('--checkpoint-allow-budget-change',resumed['command'])
        self.assertEqual(job['command'][2],'/old/snapshot/tools/run_instrumented.py')

    def test_native_checkpoint_migration_remains_outside_verified_generic_scope(self):
        saved=self.save_old();self.authorize(saved)
        self.args.task_mode='';self.args.attack_mode='PI'
        with self.assertRaises(self.cp.CheckpointError):self.fixture.load()

    def test_prompt_path_relocation_requires_identical_frozen_subtree_files(self):
        current=self.original_identity(self.args)
        bundle=Path(self.cp.__file__).resolve().parents[1]/'prompts/mmlu/prompts.yaml'
        oldroot=self.fixture.root/'maple-run-snapshots/86a8b158'
        oldfile=oldroot/'prompts/mmlu/prompts.yaml';oldfile.parent.mkdir(parents=True)
        oldfile.write_bytes(bundle.read_bytes())
        spec={'path':str(bundle),'present':True,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}
        current['prompts']={'bundle':spec}
        config=self.cp._decode(current['config']);config.update(prompt_dir='prompts/mmlu',prompt_file='prompts/mmlu/prompts.yaml')
        current['config']=self.cp._encode(config)
        saved=copy.deepcopy(current);saved['source']=self.old;saved['prompts']['bundle']['path']=str(oldfile)
        try:authorization=self.cp.evaluator_recovery_authorization(saved,current)
        except AttributeError as error:self.fail('Verified prompt relocation is missing: '+str(error))
        relocation=authorization['prompt_path_relocations']['bundle']
        self.assertEqual(relocation['relative_path'],'prompts/mmlu/prompts.yaml')
        self.assertEqual(relocation['sha256'],spec['sha256'])
        for change in ('hash','presence','external','traversal','symlink','absolute-arg','fallback-added'):
            altered=copy.deepcopy(current)
            if change=='hash':altered['prompts']['bundle']['sha256']='0'*64
            elif change=='presence':altered['prompts']['bundle']['present']=False
            elif change=='external':altered['prompts']['bundle']['path']=str(self.fixture.root/'outside.yaml')
            elif change=='traversal':altered['prompts']['bundle']['path']=str(bundle.parent/'..'/'mmlu'/'prompts.yaml')
            elif change=='symlink':
                oldfile.unlink();oldfile.symlink_to(bundle)
            elif change=='absolute-arg':
                c=copy.deepcopy(config);c['prompt_file']=str(bundle)
                altered['config']=self.cp._encode(c)
            else:altered['prompts']['new_fallback.txt']=spec
            with self.subTest(change=change),self.assertRaises(self.cp.CheckpointError):
                self.cp.evaluator_recovery_authorization(saved,altered)
            if change=='symlink':oldfile.unlink();oldfile.write_bytes(bundle.read_bytes())
