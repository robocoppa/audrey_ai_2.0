"""Exercise operator maintenance guards and restart ordering without Docker."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/ops/reembed-onbox.sh"


@pytest.fixture
def runner(tmp_path):
    binaries = tmp_path/"bin"
    binaries.mkdir()
    docker = binaries/"docker"
    docker.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['CALLS'],'a') as out: out.write(json.dumps(a)+'\\n')
if a[:2]==['container','inspect']:
    if '--format' not in a: sys.exit(1)
    if '.State.Running' in a[a.index('--format')+1]:
        print('true' if os.environ.get('ACTIVE') and a[-1]=='audrey' else 'false')
    else:
        print('MEMORY_COLLECTION=kb_memory\\nSECRET_API_TOKEN=private-marker\\nOLLAMA_HOST=http://ollama:11434')
    sys.exit(0)
if a[0]=='run' and '-m' in a:
    mount=next(arg for arg in a if arg.startswith('type=bind,src=') and ',dst=/migration' in arg)
    state=Path(mount.split('src=',1)[1].split(',dst=',1)[0])
    target=state/'reports'/Path(a[a.index('--report')+1]).name
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({'schema':1,'status':'completed' if not os.environ.get('FAIL') else 'failed'}))
    sys.exit(int(os.environ.get('FAIL','0')))
sys.exit(0)
''')
    docker.chmod(0o700)
    for name in ("data","tools","state","reports"):
        (tmp_path/name).mkdir()
    config=tmp_path/"config.yaml"
    config.write_text("text_embedding: {}")
    env=dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ['PATH'],
        CALLS=str(tmp_path/"calls"),DATA_HOST=str(tmp_path/"data"),TOOLS_HOST=str(tmp_path/"tools"),
        STATE_HOST=str(tmp_path/"state"),OUT_DIR=str(tmp_path/"reports"),CONFIG_FILE=str(config),NOTIFY="0")
    for key in ("OLLAMA_HOST","QDRANT_HOST","QDRANT_PORT","MEMORY_COLLECTION","CHAT_ARCHIVE_COLLECTION"):
        env.pop(key,None)
    return env,tmp_path


def invoke(env, args=()):
    return subprocess.run(["bash",str(SCRIPT), *args],env=env,text=True,capture_output=True,timeout=10)


def calls(path):
    return [json.loads(line) for line in (path/"calls").read_text().splitlines()]


def test_active_writer_blocks_migration_and_restart(runner):
    env,path=runner
    result=invoke({**env,"ACTIVE":"1"})
    assert result.returncode==2
    assert "Stop audrey" in result.stderr
    assert not any('-m' in call or call[0]=='compose' for call in calls(path))


def test_verified_run_mounts_sources_readonly_filters_credentials_and_restarts_in_order(runner):
    env,path=runner
    result=invoke(env)
    assert result.returncode==0,result.stderr
    seen=calls(path)
    migration=next(call for call in seen if '-m' in call)
    launches = [call for call in seen if call[0] == 'run']
    assert len(launches) == 2
    for launch in launches:
        assert 'OPENBLAS_NUM_THREADS=1' in launch
        assert 'OMP_NUM_THREADS=1' in launch
    preflight = next(call for call in launches if '-c' in call)
    assert 'import audrey.kb.reembed' in preflight[preflight.index('-c') + 1]
    assert '--maintenance-confirmed' in migration
    assert any('dst=/data,readonly' in arg for arg in migration)
    assert any('dst=/tools-data,readonly' in arg for arg in migration)
    assert 'MEMORY_COLLECTION=kb_memory' in migration
    assert not any('SECRET_API_TOKEN' in arg or 'private-marker' in arg for arg in migration)
    up=[call for call in seen if call[0]=='compose']
    assert len(up)==2 and '--wait' in up[0]
    assert up[0][-2:]==['custom-tools','audrey']
    assert up[1][-2:]==['media-worker','media-fetcher']
    assert json.loads(next((path/'reports').glob('*.json')).read_text())['status']=='completed'


def test_failed_migration_keeps_writers_stopped_and_preserves_report(runner):
    env,path=runner
    result=invoke({**env,"FAIL":"1"})
    assert result.returncode==1
    assert not any(call[0]=='compose' for call in calls(path))
    assert 'remain stopped' in result.stderr
    assert json.loads(next((path/'reports').glob('*.json')).read_text())['status']=='failed'


def test_history_scope_uses_fresh_default_state_and_forwards_only_requested_scope(runner):
    env, path = runner
    env.pop("STATE_HOST")
    state = path / "data" / "history-reembed-nomic"
    state.mkdir(mode=0o700)
    result = invoke(env, ("--scope", "history"))
    assert result.returncode == 0, result.stderr
    migration = next(call for call in calls(path) if "-m" in call)
    assert migration[migration.index("--scope") + 1] == "history"
    assert any(f"src={state},dst=/migration" in arg for arg in migration)
    assert "history-reembed-report.json" in migration[migration.index("--report") + 1]
    assert not (path / "data" / "embedding-migration-qwen4").exists()
    assert list(state.glob("reports/*-history-reembed-report.json"))


def test_invalid_scope_never_starts_docker_or_reopens_writers(runner):
    env, path = runner
    result = invoke(env, ("--scope", "files"))
    assert result.returncode == 2
    assert "scope must be all or history" in result.stderr
    assert not (path / "calls").exists()
