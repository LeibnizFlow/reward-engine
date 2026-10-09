"""Public task attachments survive product-blind construction and cross-family attack."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.clauses import normalize_criteria
from reward_engine.core.outcome.context import (
    build_context,
    requirement_context,
    requirement_scope,
)
from reward_engine.core.outcome.counterexamples import public_reference_context
from reward_engine.core.outcome.harness import measurement_context


def test_original_inputs_survive_blinding_without_promoting_modified_product_files():
    task = {'task_id':'office','prompt':'Use the supplied schema to build a workbook.',
            'documents':[{'path':'input/schema.json','content':'{"currency":"CNY_mn"}','sha256':'original'}]}
    candidate = {'artifact':{'final_response':'Done'},
                 'source_documents':[{'path':'input/schema.json','content':'ACTOR_MODIFIED_SCHEMA'}]}
    c,e = build_context(task,[candidate])
    key = 'task_input:input%2Fschema.json'
    blind, sources = requirement_context(c,e)
    assert key in sources and 'ACTOR_MODIFIED_SCHEMA' not in str(sources)
    view, refs = public_reference_context(c,e)
    assert key in refs and not view['products']
    assert all(not r.startswith('product_') for r in refs)
    clause={'id':'unit','text':'Use CNY_mn','use':'task_constraint','evidence_refs':[key],
            'source_quotes':[{'ref':key,'quote':'"currency":"CNY_mn"'}], 'conditions':[], 'alternative_group':''}
    assert normalize_criteria([{'clauses':[clause]}],c,e,required=True)
    altered=deepcopy(e)
    altered[key]['content']='{"currency":"USD"}'
    assert requirement_scope(c,e) != requirement_scope(c,altered)
    with pytest.raises(ValueError,match='exactly'):
        normalize_criteria([{'clauses':[clause]}],c,altered,required=True)


def test_fresh_review_receives_frozen_originals_and_rejects_conflicting_task_inputs():
    source={'path':'schema.json','content':'ORIGINAL_SCHEMA'}
    pack={'measurement_sources':{'task_input:schema.json':source}}
    c,e=build_context({'task_id':'office','prompt':'Use schema'})
    c,e=measurement_context(pack,c,e)
    assert e['task_input:schema.json'] == source
    assert c['reference_documents'][0]['source_kind']=='original_task_input'
    e['task_input:schema.json']={'path':'schema.json','content':'CHANGED'}
    with pytest.raises(ValueError,match='conflicts'):
        measurement_context(pack,c,e)
