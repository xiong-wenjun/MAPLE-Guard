import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests
from tools.run_recovery_plan import validate_result


def pending():
    return {'response_budget_policy':'fail_task','budget_outcomes_version':1,'task_id':'t','request_id':'j','role':'evaluator','finish_reasons':['length'],'invalid_response_type':'length','status':'pending_budget_rescore','result_key':'correct','request':{'endpoint':'http://judge/v1/chat/completions','timeout':60,'payload':{'model':'j','messages':[{'role':'user','content':'full task answer'}],'temperature':0.0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False}}}}


class BudgetEvaluatorReplayTests(unittest.TestCase):
    def directory(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        root=Path(folder.name)
        (root/'trace.summary.json').write_text(json.dumps({'response_budget_policy':'fail_task','budget_outcomes_version':1}))
        return root

    def test_validator_accepts_only_exact_matched_budget_outcome(self):
        root=self.directory()
        event={'response_budget_policy':'fail_task','budget_outcomes_version':1,'request_id':'r','task_id':'t','role':'task','invalid_response_type':'length','finish_reasons':['length']}
        row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'outcome':'budget_exhausted','is_correct':False,'budget_failures':[event]}
        (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
        (root/'api-calls.jsonl').write_text(json.dumps({**event,'invalid_for_benchmark':True})+'\n')
        status=validate_result(root,0,1,['t'])
        self.assertTrue(status['valid'])
        self.assertFalse(status['metrics_valid'], 'execution completion cannot certify unknown primary ASR')
        event['request_id']='unmatched'
        (root/'api-calls.jsonl').write_text(json.dumps({**event,'invalid_for_benchmark':True})+'\n')
        self.assertFalse(validate_result(root,0,1,['t'])['valid'])

    def test_pending_evaluator_completes_execution_without_valid_metrics(self):
        root=self.directory();p=pending()
        row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'outcome':'completed','is_correct':None,'final_answer':'full task answer','pending_evaluators':[p]}
        (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
        (root/'api-calls.jsonl').write_text(json.dumps({**{k:v for k,v in p.items() if k!='request'},'finish_reasons':['length'],'invalid_for_benchmark':True})+'\n')
        status=validate_result(root,0,1,['t'])
        self.assertTrue(status['valid'])
        self.assertFalse(status['metrics_valid'])
        self.assertTrue(status['pending_evaluation'])

    def test_validator_rejects_empty_final_attack_and_duplicate_matches(self):
        for change in ('empty_final','attack','duplicate'):
            with self.subTest(change=change):
                root=self.directory();p=pending()
                if change=='empty_final':p['invalid_response_type']='empty_final'
                if change=='attack':p['role']='attack'
                row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'outcome':'completed','is_correct':None,'pending_evaluators':[p,p] if change=='duplicate' else [p]}
                (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
                (root/'api-calls.jsonl').write_text(json.dumps({**{k:v for k,v in p.items() if k!='request'},'finish_reasons':['length'],'invalid_for_benchmark':True})+'\n')
                self.assertFalse(validate_result(root,0,1,['t'])['valid'])

    def test_validator_checks_each_event_and_call_policy_version(self):
        from maple_guard.budget_outcomes import validate_committed_events
        event={'response_budget_policy':'fail_task','budget_outcomes_version':1,'request_id':'r','task_id':'t','role':'task','invalid_response_type':'length','finish_reasons':['length']}
        row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'outcome':'budget_exhausted','is_correct':False,'budget_failures':[event]}
        call={**event,'invalid_for_benchmark':True}
        self.assertTrue(validate_committed_events([row],[call])[0])
        for target in ('event','call'):
            for field,value in (('response_budget_policy','strict'),('budget_outcomes_version',999)):
                with self.subTest(target=target,field=field):
                    altered={**event,field:value}
                    r={**row,'budget_failures':[altered if target=='event' else event]}
                    c={**call,field:value} if target=='call' else call
                    self.assertFalse(validate_committed_events([r],[c])[0])

    def test_validator_diagnostic_length_links_separately_to_completed_primary(self):
        from maple_guard.budget_outcomes import validate_committed_events
        event={'response_budget_policy':'fail_task','budget_outcomes_version':1,'request_scope':'counterfactual_mir','request_id':'d','task_id':'t','role':'task','invalid_response_type':'length','finish_reasons':['length']}
        row={'task_id':'t','response_budget_policy':'fail_task','budget_outcomes_version':1,'outcome':'completed','final_answer':'B','is_correct':True,'causal_mir_status':'budget_exhausted','memory_influence':None,'memory_caused_failure':None,'diagnostic_budget_failures':[event]}
        call={**event,'invalid_for_benchmark':True}
        self.assertTrue(validate_committed_events([row],[call])[0])
        for change in ({'request_scope':'primary'},{'role':'attack'},{'task_id':'other'},{'invalid_response_type':'empty_final'}):
            with self.subTest(change=change):
                self.assertFalse(validate_committed_events([row],[{**call,**change}])[0])
        self.assertFalse(validate_committed_events([{**row,'diagnostic_budget_failures':[event,event]}],[call])[0])
        self.assertFalse(validate_committed_events([{**row,'final_answer':''}],[call])[0])

    def test_replay_calls_only_saved_evaluator_payload_and_validates_parse(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.rescore_budget_evaluators'),'evaluator-only replay tool is missing')
        from tools.rescore_budget_evaluators import rescore_requests
        root=self.directory();trace=root/'trace.jsonl';trace.write_text(json.dumps({'task_id':'t','final_answer':'full task answer','pending_evaluators':[pending()]})+'\n')
        original=trace.read_bytes()
        reply=requests.Response();reply.status_code=200;reply._content=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"correct":true,"confidence":1,"evidence":"supported"}'}}]}).encode()
        with patch('requests.post',return_value=reply) as post, patch('maple_guard.maple_guard_core.call_chat',side_effect=AssertionError('must not rerun task agents')):
            result=rescore_requests(trace,root/'rescoring.jsonl',max_tokens=1024,max_attempts=2)
        self.assertEqual(post.call_count,1)
        sent=post.call_args.kwargs['json'];saved=pending()['request']['payload'];saved['max_tokens']=1024
        self.assertEqual(sent,saved)
        self.assertTrue(result[0]['verdict'])
        self.assertEqual(trace.read_bytes(),original)
        self.assertEqual((root/'rescoring.jsonl').stat().st_mode & 0o777,0o600)

    def test_replay_parse_failure_and_transport_do_not_become_false(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.rescore_budget_evaluators'))
        from tools.rescore_budget_evaluators import rescore_requests
        for reply in ('{"correct":"false"}',requests.Timeout('timeout')):
            with self.subTest(reply=reply):
                root=self.directory();trace=root/'trace.jsonl';trace.write_text(json.dumps({'task_id':'t','pending_evaluators':[pending()]})+'\n')
                if isinstance(reply,str):
                    r=requests.Response();r.status_code=200;r._content=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':reply}}]}).encode();kwargs={'return_value':r}
                else:kwargs={'side_effect':reply}
                with patch('requests.post',**kwargs) as post, patch('tools.rescore_budget_evaluators.time.sleep'):
                    result=rescore_requests(trace,root/'rescore.jsonl',max_tokens=1024,max_attempts=2)
                self.assertIsNone(result[0]['verdict'])
                self.assertIn(result[0]['status'],('pending_invalid_parse','pending_transport'))
                self.assertLessEqual(post.call_count,2)

    def test_replay_malformed_response_shapes_write_pending_provider_sidecar(self):
        from tools.rescore_budget_evaluators import rescore_requests
        malformed=(None,[],{'choices':None},{'choices':{}},{'choices':[None]},{'choices':[{'message':'invalid'}]},{'choices':[{'message':None}]})
        for data in malformed:
            with self.subTest(data=data):
                root=self.directory();trace=root/'trace.jsonl';out=root/'rescore.jsonl'
                trace.write_text(json.dumps({'task_id':'t','pending_evaluators':[pending()]})+'\n')
                reply=requests.Response();reply.status_code=200;reply._content=json.dumps(data).encode()
                with patch('requests.post',return_value=reply) as post:
                    result=rescore_requests(trace,out,max_tokens=1024,max_attempts=2)
                self.assertEqual(post.call_count,1)
                self.assertEqual(result[0]['status'],'pending_provider_error')
                self.assertIsNone(result[0]['verdict'])
                self.assertEqual(result[0]['attempts'][0]['error_type'],'ValueError')
                self.assertEqual(json.loads(out.read_text())['status'],'pending_provider_error')

    def test_replay_requires_strictly_larger_cap_before_network(self):
        from tools.rescore_budget_evaluators import rescore_requests
        root=self.directory();trace=root/'trace.jsonl';trace.write_text(json.dumps({'task_id':'t','pending_evaluators':[pending()]})+'\n')
        for cap in (128,256):
            with self.subTest(cap=cap),patch('requests.post') as post:
                with self.assertRaises(ValueError):
                    rescore_requests(trace,root/'rescore.jsonl',max_tokens=cap)
                post.assert_not_called()

    def test_replay_accepts_only_one_anchored_fenced_json_object(self):
        from tools.rescore_budget_evaluators import _parse
        fenced='  ```json\n{"correct":true,"confidence":1.0,"evidence":"identical"}\n```  '
        try:
            result=_parse(fenced,'correct')
        except ValueError:
            result=None
        self.assertEqual(result,(True,1.0,'identical'))
        for invalid in ('prose '+fenced, fenced+fenced, '```json\n{"correct":"true"}\n```', '{"correct":true} {"correct":false}'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                _parse(invalid,'correct')

if __name__=='__main__':unittest.main()
