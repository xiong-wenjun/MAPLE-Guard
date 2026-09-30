"""Strict external verdict rejection preserves independently observed task results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import requests

from maple_guard import budget_outcomes as budget, maple_guard_core as ep, run_mmlu, paper_metrics as paper
from tools.run_instrumented import BenchmarkResponseError, install_metrics
from tools.rescore_budget_evaluators import rescore_requests


def response(content,finish='stop'):
    reply=requests.Response();reply.status_code=200
    reply._content_consumed=True
    reply._content=json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}]}).encode()
    return reply


class SemanticPendingTests(unittest.TestCase):
    def tearDown(self):
        budget.configure_policy(SimpleNamespace(response_budget_policy='strict'))

    def args(self):
        return SimpleNamespace(response_budget_policy='fail_task',agents=2,attacker_ids=[1],
            chat_base_url='http://judge/v1',chat_model='judge',pattern_judge_base_url='http://judge/v1',
            pattern_judge_model='judge',pattern_judge_max_tokens=512,pattern_judge_timeout=60,
            rounds=3,method='no_defense_memrl',warmup_tasks=0,stream_protocol='persistent_online')

    def judge(self,raw,result_key='imitates'):
        @budget.evaluator(result_key)
        def evaluate(agent_id=0,round_idx=2):
            verdict=ep.call_chat('http://judge/v1','judge',[{'role':'user','content':'exact private answer'}],
                max_tokens=512,temperature=0.0,timeout=60)
            parsed=budget.validate_evaluator_verdict(verdict,result_key)
            return parsed[0],{result_key:parsed[0]}
        with patch('requests.post',return_value=response(raw)):
            with budget.task_scope(self.args(),'t',poison_texts={}) as audit:
                try: verdict,decision=evaluate()
                except BenchmarkResponseError as error:
                    self.fail('External semantic rejection aborted the completed task: '+str(error))
        return verdict,decision,audit

    def test_semantic_json_schema_and_confidence_errors_are_saved_exactly_unknown(self):
        for raw in ('not JSON','{"imitates":"false"}','{"imitates":true,"confidence":95,"evidence":"percent"}',
                    '{"imitates":false,"evidence":7}','{"imitates":false} {"imitates":true}'):
            with self.subTest(raw=raw):
                verdict,decision,audit=self.judge(raw)
                self.assertIsNone(verdict)
                self.assertEqual(decision['status'],'pending_invalid_verdict_rescore')
                self.assertEqual(decision['invalid_response_type'],'invalid_evaluator_verdict')
                self.assertEqual(decision['raw_verdict'],raw)
                self.assertEqual(decision['request']['payload']['messages'],[{'role':'user','content':'exact private answer'}])
                self.assertEqual(decision['request']['payload']['max_tokens'],512)
                self.assertEqual(decision['agent_id'],0);self.assertEqual(decision['round'],3)
                self.assertTrue(decision['validation_error']['message'])
                self.assertEqual(audit.pending,[decision]);self.assertFalse(audit.feedback_incomplete)
                self.assertNotIn('headers',decision['request'])

    def test_correctness_pending_marks_feedback_gap(self):
        verdict,decision,audit=self.judge('{"correct":false,"confidence":100}','correct')
        self.assertIsNone(verdict);self.assertTrue(audit.feedback_incomplete)
        outcome=budget.summarize_outcomes([dict(is_correct=None,asr_at_3=False,pending_evaluators=audit.pending,feedback_incomplete=True)])
        self.assertIsNone(outcome['accuracy']);self.assertFalse(outcome['main_table_eligible'])
        self.assertTrue(outcome['pending_correctness_evaluation'])

    def test_malformed_provider_task_defense_unscoped_and_strict_remain_fatal(self):
        for role in ('task','defense','unknown'):
            with self.subTest(role=role),budget.task_scope(self.args(),'t'),budget.role_scope(role):
                with self.assertRaises(BenchmarkResponseError):budget.validate_evaluator_verdict('invalid','imitates')
        with self.assertRaises(BenchmarkResponseError):budget.validate_evaluator_verdict('invalid','imitates')
        @budget.evaluator('imitates')
        def evaluate():
            return ep.call_chat('http://judge/v1','judge',[],max_tokens=512)
        for malformed in (None,{'choices':None},{'choices':[{'message':'bad'}]},response('').json()):
            reply=response('unused');reply._content=json.dumps(malformed).encode()
            with self.subTest(malformed=malformed),budget.task_scope(self.args(),'t'),patch('requests.post',return_value=reply):
                with self.assertRaises(BenchmarkResponseError):evaluate()
        with budget.task_scope(SimpleNamespace(response_budget_policy='strict',agents=2,attacker_ids=[1]),'t'),budget.role_scope('evaluator'):
            with self.assertRaises(BenchmarkResponseError):budget.validate_evaluator_verdict('invalid','imitates')
        with budget.task_scope(self.args(),'t'),patch('requests.post',side_effect=requests.Timeout('transport')):
            with self.assertRaises(budget.RecoverableProviderError):evaluate()

    def test_real_mmlu_pattern_pending_preserves_primary_paper_metrics(self):
        args=self.args();args.enable_pattern_judge=True;args.pattern_judge_rounds='all'
        task=ep.TaskExample('t','Question',[('A','yes'),('B','no')],'A','B',raw={})
        with budget.task_scope(args,'t') as audit,patch('requests.post',return_value=response('{"imitates":false,"confidence":99}')):
            try:
                value,decision=run_mmlu.judge_poison_pattern_imitation(task,0,2,'<ANSWER>: A',['p'],{'p':'B'},{'p':'private poison'},args)
            except BenchmarkResponseError as error:self.fail(str(error))
        self.assertIsNone(value)
        rows=[dict(task_index=0,task_id='t',is_correct=True,asr_at_3=True,outcome='completed',
            pending_evaluators=audit.pending,feedback_incomplete=False,
            paper_round_observations=paper.round_observations([0],[{0:'A'}]*3,[{0:['p']}]*3,['p'],[{0:True}]*3),
            pattern_judge_decisions=[decision])]
        observed=paper.summarize(rows,1,[0])
        self.assertEqual((observed['sr'],observed['rda_at_3'],observed['mdsr_at_3'],observed['asr_at_3']),(1.,0.,1.,1.))
        self.assertTrue(observed['paper_metrics_valid'])
        summary=budget.summarize_outcomes(rows)
        self.assertTrue(summary['main_table_eligible']);self.assertFalse(summary['pending_correctness_evaluation'])
        self.assertEqual(summary['budget_failure_counts_by_role']['evaluator'],0)
        self.assertEqual(summary['evaluator_invalid_verdict_requests'],1)
        @budget.outcome_summary('mmlu')
        def summarize(records,args,poison_indices):return {'pattern_asr':0.,'memory_conditioned_pattern_asr':0.,'mdsr':1.,'retrieval_damage_asr':0.}
        record=run_mmlu.StreamTaskRecord(trace_id='test',method=args.method,attack_capability='dmi',attack_variant='explicit',
            poison_payload='',is_poisoning_task=False,attacker_id=1,target_agent_id=0,final_answer='A',correct_answer='A',**rows[0])
        wrapped=summarize([record],args,set())
        self.assertIsNone(wrapped['pattern_asr']);self.assertIsNone(wrapped['memory_conditioned_pattern_asr'])
        self.assertEqual(wrapped['pattern_metric_audit_by_round']['3']['unknown_slots'],1)
        self.assertEqual(wrapped['pattern_metric_audit_by_round']['3']['coverage'],0.)
        self.assertEqual(wrapped['mdsr'],1.);self.assertEqual(wrapped['retrieval_damage_asr'],0.)

    def test_committed_semantic_event_matches_normal_http_call_and_replay_keeps_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=root/'api.jsonl'
            @budget.evaluator('imitates')
            def evaluate():
                raw=ep.call_chat('http://judge/v1','judge',[{'role':'user','content':'private full task'}],max_tokens=512)
                return budget.validate_evaluator_verdict(raw,'imitates')
            with budget.task_scope(self.args(),'t') as audit,patch('requests.sessions.Session.send',return_value=response('{"imitates":false,"confidence":99}')):
                try: verdict,decision=evaluate()
                except BenchmarkResponseError as error:self.fail(str(error))
            self.assertIsNone(verdict)
            # Native request events have the same request identity as journal instrumentation.
            call={**audit.requests[-1],'finish_reasons':['stop'],'invalid_for_benchmark':False}
            row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'is_correct':True,
                 'outcome':'completed','pending_evaluators':audit.pending}
            self.assertEqual(budget.validate_committed_events([row],[call]),(True,0,False))
            for altered in ({'role':'task'},{'request_id':'other'},{'messages_sha256':'wrong'},{'http_status':500}):
                with self.subTest(altered=altered):self.assertFalse(budget.validate_committed_events([row],[{**call,**altered}])[0])
            trace=root/'trace.jsonl';trace.write_text(json.dumps(row)+'\n');original=trace.read_bytes()
            with patch('requests.post',return_value=response('{"imitates":true,"confidence":1}')) as post:
                rescored=rescore_requests(trace,root/'rescore.jsonl',max_tokens=512,max_attempts=1)
            self.assertTrue(rescored[0]['verdict']);self.assertEqual(post.call_args.kwargs['json'],decision['request']['payload'])
            self.assertEqual(trace.read_bytes(),original)
            self.assertEqual((root/'rescore.jsonl').stat().st_mode & 0o777,0o600)

    def test_result_validator_distinguishes_auxiliary_pending_from_primary_validity(self):
        from tools.run_recovery_plan import validate_result
        verdict,event,audit=self.judge('{"imitates":false,"confidence":99}')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,
                 'outcome':'completed','is_correct':True,'pending_evaluators':audit.pending}
            call={**audit.requests[-1],'finish_reasons':['stop'],'invalid_for_benchmark':False}
            (root/'trace.jsonl').write_text(json.dumps(row)+'\n');(root/'api-calls.jsonl').write_text(json.dumps(call)+'\n')
            (root/'trace.summary.json').write_text(json.dumps({'response_budget_policy':'fail_task','budget_outcomes_version':1,
                'pending_evaluation':True,'pending_correctness_evaluation':False,'pending_auxiliary_evaluation':True,
                'main_table_eligible':True,'accuracy_metrics_valid':True,'asr_metrics_valid':True}))
            result=validate_result(root,0,1,['t'])
            self.assertTrue(result['valid']);self.assertTrue(result['metrics_valid'])
            self.assertTrue(result['pending_auxiliary_evaluation']);self.assertFalse(result['pending_correctness_evaluation'])
