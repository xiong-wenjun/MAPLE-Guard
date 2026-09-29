"""Explicit bounded_recent_v1 adaptation; storage never follows prompt eviction."""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
from pathlib import Path

POLICY = 'bounded_recent_v1'
HISTORY_PREFIX = 'AgentSafe permitted conversation history:\n'


class AgentSafeContextBudget:
    def __init__(self, tokenizer, context_limit, output_reserve, margin, template_kwargs=None, assets=None, content_format="string"):
        for name, value in [('context_limit', context_limit), ('output_reserve', output_reserve), ('margin', margin)]:
            if type(value) is not int or value < (0 if name == 'margin' else 1):
                raise ValueError('AgentSafe ' + name + ' must be an explicit valid integer')
        if context_limit <= output_reserve + margin:
            raise ValueError('AgentSafe context limit leaves no input budget')
        if content_format not in ('string', 'openai'):
            raise ValueError('AgentSafe content format must be string or openai')
        self.content_format = content_format
        self.tokenizer = tokenizer
        self.input_limit = context_limit - output_reserve - margin
        self.template_kwargs = dict(template_kwargs or {})
        self.metadata = {'policy': POLICY, 'context_limit': context_limit, 'output_reserve': output_reserve,
                         'margin': margin, 'input_limit': self.input_limit, 'tokenizer_assets': assets or {},
                         'template_kwargs': self.template_kwargs, 'add_generation_prompt': True,
                         'chat_template_content_format': self.content_format,
                         'target_selection': 'newest_first_observation_complete_records',
                         'reflection': 'all_junk_complete_record_batches_any_junk',
                         'tokenizer_loader': 'local_assets_checked_list_special_schema_v1'}

    def count(self, messages):
        # No estimates or network fallback: count the exact full chat template.
        normalized = copy.deepcopy(messages)
        if self.content_format == 'openai':
            # Match the configured serving parser, without changing model requests.
            for message in normalized:
                if isinstance(message.get('content'), str):
                    message['content'] = [{'type':'text', 'text':message['content']}]
        return len(self.tokenizer.apply_chat_template(normalized, tokenize=True,
            add_generation_prompt=True, **self.template_kwargs))

    def require_fit(self, messages, stage):
        count = self.count(messages)
        if count > self.input_limit:
            raise ValueError(f'AgentSafe {stage} needs {count} input tokens; budget is {self.input_limit}; no text discarded')
        return count

    @functools.lru_cache(maxsize=16384)
    def _text_size(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _with_history(base, records):
        if not records:
            return copy.deepcopy(base)
        # Restore chronological order after selecting by recency.
        return copy.deepcopy(base) + [{'role':'user', 'content':HISTORY_PREFIX +
            '\n'.join(record['text'] for record in reversed(records))}]

    def select_history(self, messages, records):
        base = copy.deepcopy(messages)
        base_count = self.require_fit(base, 'base task')
        seen, duplicates, candidates = set(), [], []
        for record in sorted(records, key=lambda r:(r['sequence'], r['memory_id']), reverse=True):
            text = record['text']
            if text in seen or (text and any(text in m['content'] for m in base)):
                duplicates.append(record['memory_id'])
                continue
            seen.add(text)
            candidates.append(record)
        # Per-record counts guide packing only. Final rendered verification below
        # is authoritative even for tokenizers with nonadditive boundary behavior.
        overhead = self.count(base + [{'role':'user', 'content':HISTORY_PREFIX}]) - base_count
        remaining = self.input_limit - base_count - overhead
        chosen, deferred = [], []
        for record in candidates:
            cost = self._text_size(record['text'] + '\n') + 1
            if cost > self.input_limit - base_count - overhead:
                self.require_fit(self._with_history(base, [record]), 'single history record ' + record['memory_id'])
            if cost <= remaining:
                chosen.append(record)
                remaining -= cost
            else:
                self.require_fit(self._with_history(base, [record]), 'single history record ' + record['memory_id'])
                deferred.append(record['memory_id'])
        prompt = self._with_history(base, chosen)
        while self.count(prompt) > self.input_limit and chosen:
            record = chosen[-1]
            self.require_fit(self._with_history(base, [record]), 'single history record ' + record['memory_id'])
            deferred.append(chosen.pop()['memory_id'])
            prompt = self._with_history(base, chosen)
        count = self.require_fit(prompt, 'selected history')
        audit = {**self.metadata, 'input_tokens':count, 'base_input_tokens':base_count,
                 'selected_ids':[r['memory_id'] for r in chosen], 'deferred_ids':deferred,
                 'duplicate_ids':duplicates, 'stored_record_count':len(records), 'storage_evictions':0}
        return prompt, audit

    def batch_junk(self, build, junk):
        self.require_fit(build([]), 'reflection candidate and criteria')
        batches, current = [], []
        for record in junk:
            trial = current + [record]
            if self.count(build(trial)) <= self.input_limit:
                current = trial
                continue
            self.require_fit(build([record]), 'single junk record')
            if current:
                batches.append(build(current))
            current = [record]
        if current or not batches:
            batches.append(build(current))
        return batches


@functools.lru_cache(maxsize=4)
def _load_tokenizer(path):
    from transformers import AutoTokenizer
    source = Path(path)
    if not source.is_dir():
        raise ValueError('AgentSafe tokenizer must name an existing local asset directory')
    config = json.loads((source / 'tokenizer_config.json').read_text())
    extra = config.get('extra_special_tokens')
    options = {}
    expected_ids = {}
    if isinstance(extra, list):
        # New asset schema uses a list; installed transformers 4 expects a map.
        # These tokens already exist in tokenizer.json. Validate every declared
        # token before suppressing only redundant Python attribute registration.
        native = json.loads((source / 'tokenizer.json').read_text())
        added = {item['content']:item for item in native['added_tokens']}
        if any(not isinstance(token, str) or token not in added or not added[token]['special'] for token in extra):
            raise ValueError('AgentSafe special-token schema is not backed by native special token IDs')
        expected_ids = {token:item['id'] for token,item in added.items()}
        options['extra_special_tokens'] = {}
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False, **options)
    if any(tokenizer.convert_tokens_to_ids(token) != index for token,index in expected_ids.items()):
        raise ValueError('AgentSafe tokenizer loader changed native special token IDs')
    assets = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.iterdir()) if p.is_file() and
              (p.name.startswith('tokenizer') or p.name in {'chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'})}
    if not assets or not getattr(tokenizer, 'chat_template', None):
        raise ValueError('AgentSafe tokenizer has no pinned assets or chat template')
    return tokenizer, assets


def make_context_budget(args, kind):
    policy = getattr(args, 'agentsafe_context_policy', 'full')
    if policy == 'full':
        return None
    if policy != POLICY:
        raise ValueError('Unknown AgentSafe context policy: ' + str(policy))
    tokenizer, assets = _load_tokenizer(getattr(args, 'agentsafe_' + kind + '_tokenizer', ''))
    limit = getattr(args, 'agentsafe_' + kind + '_context_limit', None)
    if kind == 'target':
        reserve = int(os.environ.get('CHAT_MAX_TOKENS', getattr(args, 'chat_max_tokens', 128)))
        thinking = bool(getattr(args, 'disable_chat_thinking', False)) or os.getenv('CHAT_DISABLE_THINKING') == '1'
    else:
        reserve = getattr(args, 'full_judge_max_tokens', 4096)
        thinking = bool(getattr(args, 'disable_chat_thinking', False))
    return AgentSafeContextBudget(tokenizer, limit, reserve, getattr(args, 'agentsafe_context_margin', 128),
                                 {'enable_thinking':False} if thinking else {}, assets,
                                 getattr(args, 'agentsafe_' + kind + '_content_format', 'string'))
