"""Reducer-only navigation projection and reversible exact text sharing."""

import json
from collections import Counter
from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint

VERSION = 'outcome-reducer-context/2026-09-15-v1'
TEXT_KEY = '$reducer_text'
TEXT_INSTRUCTION = (
    'Objects with only $reducer_text refer to the exact string in shared_texts. '
    'Read that string at every occurrence; sharing text does NOT merge identities, '
    'scope, evidence, priority or decisions. Return ordinary literal output strings.'
)
NAVIGATION_INSTRUCTION = (
    'Memory details_ref retrieves a complete historical record, not current evidence. '
    'The method body is indexed, not absent. Its observed changes are not causal proof. '
    'Navigation-only refs may be read but cannot support final standards/gaps; cite the '
    'original Task/product/check handles after verifying them. Unread material is unknown. '
    'Revision decisions are member proposals, not accepted removals. Audit every rewrite, '
    'deletion and pending original against Task. Identical retained parts share selectors; '
    'one Reduce decision expands to every alias without losing the original coverage.'
)


def byte_size(value):
    """Match the actual core and transport JSON accounting, including separators."""
    return len(json.dumps(value, ensure_ascii=False).encode())


def context_view(context, evidence):
    view, sources = deepcopy(context), dict(evidence)
    for index, memory in enumerate(view.get('memories', [])):
        if not isinstance(memory, dict) or not isinstance(memory.get('value'), dict):
            continue  # Unknown historical shapes remain intact.
        if ('details_ref' in memory or 'navigation_only' in memory
                or 'method_record' in memory['value']):
            continue  # Never overwrite an existing field with navigation metadata.
        original = deepcopy(memory)
        ref = 'navigation:reducer:memory:' + fingerprint(original)
        if ref in sources and sources[ref] != original:
            raise ValueError('Reducer memory reference collision')
        value = deepcopy(memory['value'])
        method = value.pop('method', None)
        if method is not None:
            value['method_record'] = {'details_ref': ref, 'field': 'value.method',
                                      'bytes': byte_size(method), 'sha256': fingerprint(method)}
        key = memory.get('key')
        task = context.get('task', {}).get('prompt')
        if isinstance(key, str) and isinstance(task, str) and task and key.startswith(task):
            memory['key'] = {'task_ref': 'task', 'suffix': key[len(task):]}
        memory.update(value=value, details_ref=ref, navigation_only=True)
        if byte_size(memory) >= byte_size(original):
            view['memories'][index] = original
        else:
            sources[ref] = original
    view['available_evidence_refs'] = list(sources)
    return view, sources


def share_texts(request):
    """Intern repeated long strings, without changing any scope or unique text."""
    counts = Counter()
    collision = False

    def count(value):
        nonlocal collision
        if isinstance(value, str) and len(value.encode()) >= 160:
            counts[value] += 1
        elif isinstance(value, dict):
            if TEXT_KEY in value:
                collision = True
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)

    # Protocol instructions and response schema remain literal and familiar.
    fields = {k: v for k, v in request.items() if k not in {
        'shared_instruction', 'instruction', 'response', 'evidence_access', 'reward_revision_advice'}}
    count(request)  # Reserved literals in excluded fields must also be safe.
    counts.clear()
    count(fields)
    if collision or 'shared_texts' in request or 'text_reference_instruction' in request:
        return deepcopy(request)
    refs = {s: f't{i}' for i, (s, n) in enumerate(counts.items()) if n > 1}
    if not refs:
        return deepcopy(request)

    def encode(value):
        if isinstance(value, str) and value in refs:
            return {TEXT_KEY: refs[value]}
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return deepcopy(value)

    result = deepcopy(request)
    result.update({k: encode(v) for k, v in fields.items()})
    result.update(shared_texts={ref: text for text, ref in refs.items()},
                  text_reference_instruction=TEXT_INSTRUCTION)
    return result if byte_size(result) < byte_size(request) else deepcopy(request)


def expand_texts(request):
    """Audit/test inverse; never inflate the request before sending it to the model."""
    result = deepcopy(request)
    if 'shared_texts' not in result:
        return result
    texts = result.pop('shared_texts')
    result.pop('text_reference_instruction')

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {TEXT_KEY}:
                return texts[value[TEXT_KEY]]
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    return expand(result)


def resolve_output_texts(request, answer):
    """Restore exact reducer text aliases echoed in a final response."""
    texts = request.get('shared_texts', {})
    resolved = []

    def expand(value, path):
        if isinstance(value, dict):
            if set(value) == {TEXT_KEY}:
                ref = value[TEXT_KEY]
                if not isinstance(ref, str) or not isinstance(texts.get(ref), str):
                    raise ValueError('Reducer response contains an unknown text reference')
                resolved.append({'path': path, 'ref': ref, 'sha256': fingerprint(texts[ref])})
                return texts[ref]
            return {key: expand(item, path + '/' + str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item, path + '/' + str(index)) for index, item in enumerate(value)]
        return deepcopy(value)

    return expand(answer, ''), resolved
