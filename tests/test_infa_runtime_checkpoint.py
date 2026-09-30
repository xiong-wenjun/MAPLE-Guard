"""Native PI/TA comparison checkpoints use the explicit runtime codec once."""
import contextlib
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from maple_guard import infa_memlink_eval as runner, maple_guard_core as ep, task_checkpoint as checkpoint
from maple_guard.memory_backend import MemoryBackendBundle
from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
from evaluate.defense_methods.piguard import PIGuardDetector
from tests.test_piguard import LogitModel, Tokenizer


class InfaRuntimeCheckpointTests(unittest.TestCase):
    def test_native_pi_piguard_checkpoint_resume_preserves_runtime_and_metrics(self):
        self.equivalent('PI')

    def test_native_ta_piguard_checkpoint_resume_preserves_runtime_and_metrics(self):
        self.equivalent('TA')

    def equivalent(self, mode):
        initialize = PIGuardDetector.__init__
        def initialize_stub(detector,args):
            initialize(detector,args,tokenizer_loader=lambda *a,**kw:Tokenizer(),
                       model_loader=lambda *a,**kw:LogitModel([[4.,1.]]))
        evaluations=ModuleType('utils.evaluation_utils')
        evaluations.evaluate_output_prompted=lambda *a,**kw:{'eval':'unsucc'}
        outputs=[]
        with tempfile.TemporaryDirectory() as temporary:
            for interrupted in (False,True):
                folder=Path(temporary)/str(interrupted);folder.mkdir()
                method='piguard_lifecycle'
                with patch('sys.argv',['runner','--attack-mode',mode,'--chat-base-url','http://unused/v1',
                        '--output-root',str(folder),'--baseline-state-path',str(folder/'state.json'),
                        '--strict-comparison','--agents','2','--rounds','3','--memory-backend','memrl',
                        '--task-checkpoint-dir',str(folder/'checkpoints')]):args=runner.parse_args()
                args.memory_store_dir=str(folder/'store');Path(args.memory_store_dir).mkdir()
                args.memory_run_id='native-checkpoint-test'
                args.write_final_json=False;args.progress_every=0
                cases=[]
                for index in range(3):
                    case=dict(source_bundle_id=f'native{index}',question=f'Q{index}?\nA. yes\nB. no',
                              correct_answer='A',wrong_answer=['B'],adj_matrix=[[0,1],[1,0]],
                              attacker_idxes=[1],system_prompts=['system','system'])
                    if mode=='TA':
                        case.update(user_instruction=f'Find native{index}',available_tool_names=['User','Attack'],
                                    available_tool_descriptions='native tools',agent_scratchpad='external observation',
                                    attacker_instruction='Attack',user_tools='User',attacker_tools=['Attack'])
                    cases.append(case)
                calls=[]
                def generate(base,model,messages,**kw):
                    runtime=runner.current_runtime(method)
                    self.assertIs(type(runtime),ComparisonRuntime)
                    index=int(runtime.task_id[-1]);calls.append(index)
                    if interrupted and not args.resume_task_checkpoint and index==1:
                        Path(args.memory_store_dir,'partial').write_text('uncommitted')
                        raise TimeoutError('injected native task failure')
                    self.assertFalse(Path(args.memory_store_dir,'partial').exists())
                    reason=str(random.random())
                    return f'Thought: safe native evidence {reason}.\nFinal Answer: complete original task safely.' if mode=='TA' else f'Reason: {reason}\n<ANSWER>: A'
                def retrieve(task,agent,method,topology,private,shared,*unused):
                    runtime=runner.current_runtime(method)
                    entry=private[agent][-1] if private[agent] else ep.create_benign_memory(task,agent,'A',True,'seed')
                    if not private[agent]:private[agent].append(entry)
                    if entry.memory_id not in runtime.ledger:
                        runtime.observe_ingress(entry,'user_history','private',agent)
                    runtime.prepare_entries([entry])
                    selected,decisions=runtime.filter_entries([entry],agent)
                    return selected,[],[]
                def commit(ep,task,case,round_index,responses,method,args,private,shared,*unused):
                    runtime=runner.current_runtime(method)
                    decisions=[];written=[]
                    for agent,text in responses.items():
                        entry=ep.create_benign_memory(task,agent,'A',True,'native')
                        entry.memory_id=f'{method}_{task.task_id}_r{round_index}_a{agent}'
                        entry.experience=text
                        runtime.observe_ingress(entry,'peer_message','private',agent)
                        allowed,items=runtime.admit(entry,'private',agent);decisions.extend(items)
                        if allowed:private[agent].append(entry);written.append(entry.memory_id)
                    return decisions,written
                with contextlib.ExitStack() as context:
                    context.enter_context(patch.dict(os.environ,MEMOS_BASE_PATH=str(folder/'memos')))
                    context.enter_context(patch.dict('sys.modules',{'utils.evaluation_utils':evaluations}))
                    context.enter_context(patch.object(PIGuardDetector,'__init__',initialize_stub))
                    # Real runtime/detector/codecs; only transport/storage boundaries are offline.
                    for obj,name,function in ((ep,'create_memory_backend_bundle',lambda args,entry:MemoryBackendBundle(args,entry)),
                            (ep,'retrieve_for_agent',retrieve),(ep,'call_chat',generate),
                            (runner,'first_prompt',lambda case,agent,args,rng,method:f'TASK:{case["source_bundle_id"]}:{rng.random()}'),
                            (runner,'regen_prompt',lambda *a:'continue native task'),(runner,'commit_round_memory',commit)):
                        context.enter_context(patch.object(obj,name,side_effect=function))
                    context.enter_context(patch('evaluate.defense_methods.comparison_runtime.time.perf_counter',return_value=1.0))
                    random.seed(42)
                    try:
                        if interrupted:
                            with self.assertRaisesRegex(TimeoutError,'injected native task failure'):
                                runner.run_one_method(ep,cases,method,args)
                            args.resume_task_checkpoint=True
                            vars(args).pop('_full_baseline_runtime',None)
                        runner.run_one_method(ep,cases,method,args)
                    except checkpoint.CheckpointError as error:
                        self.fail('Native ComparisonRuntime checkpoint failed: '+str(error))
                    args.method=method
                    saved=checkpoint.load_checkpoint(args)
                    runtime=args._full_baseline_runtime
                    self.assertIs(type(runtime),ComparisonRuntime)
                    self.assertIsNone(saved['stream_state']['official_defense_state'])
                    captured=saved['_runtime']['bundle']['comparison']
                    self.assertEqual(captured['kind'],'ComparisonRuntime')
                    self.assertEqual(captured['guard']['class'],'PIGuardDetector')
                    self.assertEqual(captured['guard']['attributes']['_counters'],runtime.guard._counters)
                    self.assertGreater(runtime.guard._counters['model_calls'],0)
                    self.assertEqual(len(runtime.ledger),8+8+8+2)
                    self.assertTrue(runtime.selected and runtime.consumed and runtime.received)
                    # Object references in selected/consumed must share the saved memory graph.
                    fresh=MemoryBackendBundle(args,ep.MemoryEntry)
                    checkpoint.restore_bundle(fresh,saved)
                    restored=args._full_baseline_runtime
                    memory_objects={id(entry) for entries in fresh.private_memories.values() for entry in entries}
                    self.assertTrue(all(id(entry) in memory_objects for entries in restored.selected.values() for entry in entries))
                    self.assertTrue(all(id(entry) in memory_objects for entries in restored.consumed.values() for entry in entries.values()))
                    self.assertEqual(restored.guard._counters,runtime.guard._counters)
                    self.assertEqual(restored.ledger,runtime.ledger)
                    self.assertEqual(restored._pi_cache,runtime._pi_cache)
                    self.assertEqual(restored.received,runtime.received)
                    trace=[json.loads(line) for line in (folder/f'{method}.trace.jsonl').read_text().splitlines()]
                    self.assertEqual(len(trace),3)
                    self.assertEqual(calls.count(0),8)
                    self.assertEqual(calls.count(1),9 if interrupted else 8)
                    self.assertEqual(calls.count(2),8)
                    outputs.append(dict(trace=trace,metrics=saved['stream_state']['metrics'],
                                        guard=copy.deepcopy(runtime.guard._counters),ledger=copy.deepcopy(runtime.ledger)))
            self.assertEqual(outputs[0],outputs[1])
