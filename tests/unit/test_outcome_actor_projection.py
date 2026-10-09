"""The Actor sees fallible standards and source leads, not a hidden answer key."""

from reward_engine.core.outcome.loop import actor_reward_prompt


def test_reward_projection_passes_sources_without_private_payload_or_tool_quota():
    pack = {'criteria': [{'standard': 'Explain the estimate', 'anchors': {'full': 'Support the conclusion'}}],
            'measurement_sources': {'s1': {'url': 'https://example.org/manual', 'text': 'SOURCE_BODY'},
                                    's2': {'url': 'https://example.org/manual'},
                                    's3': {'url': 'file:///private/private.json'}},
            'private_grade': 'GRADER_SECRET', 'historical_user_answers': 'OLD_USER_REPLY'}
    prompt = actor_reward_prompt(pack)
    assert 'Explain the estimate' in prompt and 'Support the conclusion' in prompt
    assert prompt.count('https://example.org/manual') == 1
    assert all(x not in prompt for x in ['SOURCE_BODY', 'GRADER_SECRET', 'OLD_USER_REPLY', 'file:///'])
    assert 'fallible guidance, not proof' in prompt and 'no tool-call quota' in prompt
    assert 'does not authorize implementation' in prompt and 'sound derivation can suffice' in prompt


def test_source_leads_are_bounded_and_never_transfer_old_user_requirement():
    pack = {'criteria': [{'origin': 'user', 'standard': 'PRIOR_USER_ANSWER',
                          'anchors': {'full': 'ALSO_PRIVATE'}}],
            'measurement_sources': {str(i): {'url': f'https://example.org/{i}'} for i in range(20)}}
    prompt = actor_reward_prompt(pack)
    assert 'PRIOR_USER_ANSWER' not in prompt and 'ALSO_PRIVATE' not in prompt
    assert 'clarify unknowns before committing' in prompt
    assert sum(prompt.count(f'https://example.org/{i}\n') for i in range(20)) <= 6
    assert 'https://example.org/6' not in prompt
