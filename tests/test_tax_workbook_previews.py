"""Tax previews project the same owner-authorized immutable workbook revision."""
from dataclasses import replace

from test_tax_review import state as state
from test_tax_workbook_routes import _saved


async def test_preview_retained_workbook_sources_without_download_or_model_calls(state, tmp_path):
    saved = await _saved(state, tmp_path)
    prepared = await state.client.post(state.path + '/exports', json={'expected_revision': saved['revision']})
    path = prepared.json()['export']['download_url'].removesuffix('/download') + '/preview'
    response = await state.client.get(path)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview['format'] == 'xlsx' and preview['kind'] == 'table'
    assert [sheet['name'] for sheet in preview['sheets']] == ['Overview', 'Facts', 'Evidence', 'Calculations', 'Unresolved items']
    evidence = next(sheet for sheet in preview['sheets'] if sheet['name'] == 'Evidence')
    assert any('Dated official supporting passage.' in str(row) for row in evidence['rows'])
    assert any(state.metadata['source_url'] in str(row) for row in evidence['rows'])
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-tax-case-revision'] == str(saved['revision'])
    assert 'content-disposition' not in response.headers
    current = (await state.client.get(state.path)).json()['case']
    assert current['revision'] == saved['revision']
    assert current['evidence'][0]['passage'] == 'Dated official supporting passage.'
    state.current[0] = replace(state.owner, auth_method='personal_token', scopes=frozenset({'account:read'}))
    assert (await state.client.get(path)).status_code == 403
    state.current[0] = await state.store.resolve_external_identity(provider='test', subject='preview-other',
        email='other@example.com', display_name='Other', role='user', auth_method='test')
    assert (await state.client.get(path)).status_code == 404
    state.current[0] = state.owner
    await state.store.conversations.delete(user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id)
    assert (await state.client.get(path)).status_code == 404


async def test_old_preview_stays_at_saved_revision(state, tmp_path):
    saved = await _saved(state, tmp_path)
    prepared = await state.client.post(state.path + '/exports', json={'expected_revision': saved['revision']})
    path = prepared.json()['export']['download_url'].removesuffix('/download') + '/preview'
    changed = await state.client.post(state.path + '/actions', json={
        'operation': 'change_year', 'expected_revision': saved['revision'], 'data': {'tax_year': 2024},
    })
    assert changed.status_code == 200, changed.text
    preview = (await state.client.get(path)).json()
    overview = dict(preview['sheets'][0]['rows'])
    assert overview['Tax year'] == '2025'
    assert overview['Case revision'] == str(saved['revision'])
