"""Regression coverage for the opt-in bounded AgentSafe prompt adaptation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        return list(self.render(messages))

    @staticmethod
    def render(messages):
        return '<chat>' + ''.join(m['role'] + ':' + m['content'] + '<end>' for m in messages) + '<assistant>'

    def encode(self, text, add_special_tokens=False):
        return list(text)


class ContextBudgetTests(unittest.TestCase):
    def budget(self, limit=500):
        self.assertIsNotNone(importlib.util.find_spec('evaluate.defense_methods.agentsafe_context'),
                             'AgentSafe has no bounded complete-record context policy')
        from evaluate.defense_methods.agentsafe_context import AgentSafeContextBudget
        return AgentSafeContextBudget(CharacterTokenizer(), limit, 40, 10, {'enable_thinking': False})

    def test_bounded_history_keeps_recent_records_and_full_cache(self):
        budget = self.budget()
        base = [{'role': 'system', 'content': 'PRIVATE TASK ROLE'}, {'role': 'user', 'content': 'CURRENT TASK'}]
        records = [{'memory_id':str(i), 'text':str(i) + 'x' * 100, 'sequence':i} for i in range(20)]
        original = copy.deepcopy(records)
        prompt, audit = budget.select_history(base, records)
        self.assertLessEqual(budget.count(prompt) + 50, 500)
        self.assertEqual(prompt[:2], base)
        self.assertEqual(records, original)
        self.assertIn('19', audit['selected_ids'])
        self.assertIn('0', audit['deferred_ids'])
        self.assertEqual(set(audit['selected_ids'] + audit['deferred_ids']), {str(i) for i in range(20)})
        self.assertEqual(budget.select_history(base, records), (prompt, audit))

    def test_history_deduplicates_text_already_in_input_and_across_ids(self):
        budget = self.budget(1000)
        text = 'a complete routed peer message'
        base = [{'role':'user', 'content':'Task. Agent 1: ' + text}]
        records = [{'memory_id':'route', 'text':text, 'sequence':1},
                   {'memory_id':'memory', 'text':text, 'sequence':2},
                   {'memory_id':'a', 'text':'other full content', 'sequence':3},
                   {'memory_id':'b', 'text':'other full content', 'sequence':4}]
        prompt, audit = budget.select_history(base, records)
        self.assertEqual(sum(m['content'].count(text) for m in prompt), 1)
        self.assertEqual(sum(m['content'].count('other full content') for m in prompt), 1)
        self.assertEqual(set(audit['duplicate_ids']), {'route', 'memory', 'a'})

    def test_oversized_task_and_record_fail_before_generation(self):
        budget = self.budget(200)
        with self.assertRaisesRegex(ValueError, 'base task'):
            budget.select_history([{'role':'user', 'content':'x' * 200}], [])
        with self.assertRaisesRegex(ValueError, 'single history record'):
            budget.select_history([{'role':'user', 'content':'task'}],
                                  [{'memory_id':'too-long', 'text':'z' * 300, 'sequence':0}])

    def test_reflection_batches_include_every_junk_record_once(self):
        budget = self.budget(420)
        junk = [{'text':str(i) + 'x' * 65, 'owner':0} for i in range(7)]
        def build(items):
            return [{'role':'system', 'content':'review'}, {'role':'user', 'content':json.dumps({'candidate':'ok','junk':items})}]
        batches = budget.batch_junk(build, junk)
        self.assertGreater(len(batches), 1)
        self.assertEqual([j for messages in batches for j in json.loads(messages[1]['content'])['junk']], junk)
        for messages in batches:
            self.assertLessEqual(budget.count(messages) + 50, 420)
        self.assertEqual(budget.batch_junk(build, []), [build([])])
        with self.assertRaisesRegex(ValueError, 'single junk record'):
            budget.batch_junk(build, [{'text':'z' * 500,'owner':0}])

    def test_whole_render_verification_handles_nonadditive_tokenizer(self):
        budget = self.budget(500)
        original = budget.tokenizer.apply_chat_template
        budget.tokenizer.apply_chat_template = lambda messages, **kwargs: original(messages, **kwargs) * (2 if len(messages)>1 else 1)
        prompt, audit = budget.select_history([{'role':'user','content':'task'}],
            [{'memory_id':str(i),'text':str(i) + 'x'*80,'sequence':i} for i in range(5)])
        self.assertLessEqual(budget.count(prompt) + 50, 500)
        self.assertGreater(len(audit['deferred_ids']), 0)

    def test_tokenizer_error_never_falls_back_to_estimation(self):
        budget = self.budget()
        def broken(*a, **k):
            raise RuntimeError('broken template')
        budget.tokenizer.apply_chat_template = broken
        with self.assertRaisesRegex(RuntimeError, 'broken template'):
            budget.select_history([{'role':'user','content':'task'}], [])


class GuardContextTests(unittest.TestCase):
    def test_batched_reflection_ors_all_judgments_and_preserves_junk(self):
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from test_agentsafe_full import FakeJudge
        helper = ContextBudgetTests()
        budget = helper.budget(2200)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'policy'; c=Path(folder)/'criteria'
            p.write_text(json.dumps({'identities':{'0':'Agent 0'}})); c.write_text('["criterion"]')
            args=SimpleNamespace(agentsafe_policy_file=str(p),agentsafe_criteria_file=str(c),
                agentsafe_threshold=0.5,agentsafe_context_policy='bounded_recent_v1')
            judge=FakeJudge(level=1)
            guard=AgentSafeFull(args,judge,lambda t:[1.,0.],judge_budget=budget)
            for i in range(6):
                judge.identity_valid=False
                guard.admit('junk'+str(i),str(i)+'j'*350,0,0,{})
            judge.identity_valid=True
            guard.admit('good','good memory',0,0,{})
            before=copy.deepcopy(guard.junk)
            calls=[]
            def reflection(messages):
                payload=json.loads(messages[1]['content'])
                calls.append(payload)
                return json.dumps({'junk':any(j['text'].startswith('4') for j in payload['junk'])})
            guard.judge=reflection
            decisions=guard.review([],0)
            self.assertEqual(decisions[0]['action'],'quarantine')
            self.assertGreater(decisions[0]['details']['reflection_batches'],1)
            self.assertEqual([j for call in calls for j in call['junk']],
                             [{'text':r['text'],'owner':r['owner']} for r in before['0'].values()])
            self.assertTrue(set(before['0']).issubset(guard.junk['0']))

    def test_first_observation_recency_survives_reads_and_restore(self):
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from test_agentsafe_full import FakeJudge
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'policy'; c=Path(folder)/'criteria'
            p.write_text(json.dumps({'identities':{'0':'Agent 0'}})); c.write_text('["criterion"]')
            args=SimpleNamespace(agentsafe_policy_file=str(p),agentsafe_criteria_file=str(c),agentsafe_threshold=0.5)
            judge=FakeJudge(level=1)
            guard=AgentSafeFull(args,judge,lambda t:[1.,0.])
            guard.admit('old','old',0,0,{})
            guard.admit('new','new',0,0,{})
            guard.read('old','old',0,0,{})
            history={r['memory_id']:r for r in guard.history(0)}
            self.assertIn('sequence',history['old'],'first observation order is not retained')
            self.assertLess(history['old']['sequence'],history['new']['sequence'])
            restored=AgentSafeFull(args,judge,lambda t:[1.,0.]); restored.load_state_dict(guard.state_dict())
            self.assertEqual(restored.history(0),guard.history(0))

if __name__ == '__main__':
    unittest.main()

class TokenizerAssetTests(unittest.TestCase):
    def test_list_special_tokens_are_loaded_without_changing_token_ids_or_template(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from evaluate.defense_methods.agentsafe_context import _load_tokenizer
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)
            tokenizer=Tokenizer(WordLevel({'[UNK]':0,'<extra>':1},unk_token='[UNK]'))
            tokenizer.add_special_tokens(['<extra>'])
            tokenizer.save(str(source/'tokenizer.json'))
            (source/'tokenizer_config.json').write_text(json.dumps({
                'tokenizer_class':'PreTrainedTokenizerFast','unk_token':'[UNK]',
                'extra_special_tokens':['<extra>'], 'chat_template':"{{ messages[0]['content'] }}<extra>"}))
            try:
                loaded, assets=_load_tokenizer(folder)
            except AttributeError as exc:
                self.fail('released tokenizer list schema must retain native token IDs: ' + str(exc))
            self.assertEqual(loaded.convert_tokens_to_ids('<extra>'),1)
            self.assertEqual(loaded.apply_chat_template([{'role':'user','content':'<extra>'}],tokenize=False),'<extra><extra>')
            self.assertIn('tokenizer.json',assets)

class RuntimeContextTests(unittest.TestCase):
    def test_runtime_sends_bounded_prompt_and_audits_without_removing_guard_records(self):
        from unittest.mock import patch
        from evaluate.defense_methods.full_runtime import FullRuntime
        budget=ContextBudgetTests().budget(500)
        history=[{'memory_id':str(i),'text':str(i)+'x'*100,'owner':0,'sequence':i} for i in range(20)]
        guard=SimpleNamespace(history=lambda *a,**k:copy.deepcopy(history),
                              read=lambda *a,**k:(True,{}),provenance={})
        args=SimpleNamespace(method='agentsafe_full',agentsafe_context_policy='bounded_recent_v1')
        with patch('evaluate.defense_methods.agentsafe_context.make_context_budget',return_value=budget):
            runtime=FullRuntime(args,guard=guard)
        seen=[]
        runtime.generate(0,[{'role':'user','content':'CURRENT TASK'}],lambda m:seen.append(m) or 'answer')
        self.assertLessEqual(budget.count(seen[0]),budget.input_limit)
        audit=next(d for d in runtime.pending if d['reason']=='agentsafe_context_budget')
        self.assertGreater(len(audit['details']['deferred_ids']),0)
        self.assertEqual(len(history),20)

class WholeRecordFailureTests(unittest.TestCase):
    def test_single_record_overflow_cannot_hide_behind_segment_estimate(self):
        budget=ContextBudgetTests().budget(500)
        original=budget.tokenizer.apply_chat_template
        budget.tokenizer.apply_chat_template=lambda messages,**kwargs:original(messages,**kwargs)*(10 if messages[-1]['content'].endswith('x'*100) else 1)
        with self.assertRaisesRegex(ValueError,'single history record'):
            budget.select_history([{'role':'user','content':'task'}],
                [{'memory_id':'long-render','text':'x'*100,'sequence':0}])

class DeployedContentFormatTests(unittest.TestCase):
    def test_openai_block_counting_matches_template_branch_without_mutating_request(self):
        from evaluate.defense_methods.agentsafe_context import AgentSafeContextBudget
        class BlockSensitiveTokenizer(CharacterTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                self.received=copy.deepcopy(messages)
                text=''
                for m in messages:
                    content=m['content']
                    text += content.strip() if isinstance(content,str) else ''.join(x['text'].strip()+' ' for x in content)
                return list(text)
        tokenizer=BlockSensitiveTokenizer()
        budget=AgentSafeContextBudget(tokenizer,500,40,10)
        self.assertIn('chat_template_content_format',budget.metadata,
                      'deployed content normalization is not part of the context protocol')
        budget=AgentSafeContextBudget(tokenizer,500,40,10,content_format='openai')
        messages=[{'role':'system','content':'system.'},{'role':'user','content':'task.'}]
        original=copy.deepcopy(messages)
        self.assertEqual(budget.count(messages),len('system. task. '))
        self.assertEqual(messages,original)
        self.assertEqual(tokenizer.received[0]['content'],[{'type':'text','text':'system.'}])
        self.assertEqual(budget.metadata['chat_template_content_format'],'openai')
        with self.assertRaisesRegex(ValueError,'content format'):
            AgentSafeContextBudget(tokenizer,500,40,10,content_format='guessed')
