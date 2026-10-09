"""Rebuild both main experiment reports from portable accepted evidence."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from pathlib import Path
from urllib.parse import quote

from .evidence import Evidence, sha256

EXPERIMENTS = {'workbuddy_office': (50, 150, 450), 'claw_eval': (104, 312, 936)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_rows(data):
    expected_tasks, expected_rounds, expected_scores = EXPERIMENTS[data['experiment']]
    rows = data['rounds']
    keys = [(r['task'], r['round']) for r in rows]
    if len(rows) != expected_rounds or len(set(keys)) != len(keys):
        raise ValueError('Duplicate or missing task-round records')
    tasks = {r['task'] for r in rows}
    if len(tasks) != expected_tasks or set(keys) != {(t, n) for t in tasks for n in (1, 2, 3)}:
        raise ValueError('Each task must have exactly three rounds')
    for row in rows:
        if row['selected_arm'] not in 'ABC' or row['selected_score'] != row[row['selected_arm']]:
            raise ValueError('Selected score does not match the recorded arm')
        for arm in 'ABC':
            value = row[arm]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Missing or invalid accepted score')
    return {'tasks': len(tasks), 'task_rounds': len(rows), 'valid_scores': expected_scores}


def rebuild(data_root: Path, output: Path, experiment: str):
    if output.exists():
        raise FileExistsError('Use a new report output directory')
    evidence = Evidence(data_root)
    source = data_root / experiment / 'experiment.json'
    data = json.loads(source.read_text())
    evidence.original_hash(source)
    totals = validate_rows(data)
    for name in data['required_evidence']:
        if name not in evidence.records:
            raise ValueError('Missing evidence registration: ' + name)
        if evidence.original_hash(evidence.resolve(name)) is None:
            raise ValueError('Missing accepted evidence: ' + name)
    rows = sorted(data['rounds'], key=lambda r: (r['task'], r['round']))
    metrics = {str(n): {arm: statistics.mean(r[arm] for r in rows if r['round'] == n)
                        for arm in ('A', 'B', 'C', 'selected_score')} for n in (1, 2, 3)}
    output.mkdir(parents=True)
    result = {**data, 'metrics': metrics, 'counts': totals, 'score_scale': [0, 100],
              'primary_result': 'round_3', 'public_input_sha256': sha256(source)}
    write_json(output / 'results.json', result)
    if data.get('task_details'):
        write_json(output / 'task_details.json', data['task_details'])
    columns = ['task', 'name', 'round', 'A', 'B', 'C', 'selected_arm', 'selected_score']
    if experiment == 'claw_eval':
        columns.append('A_mean_round1')
    for filename, selected in [('rounds.csv', rows), ('scores.csv', [r for r in rows if r['round'] == 3])]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader(); writer.writerows(selected)
    title = 'WorkBuddy Bench Office · 50 题' if experiment == 'workbuddy_office' else 'Claw-Eval · 104 题'
    lines = [f'# {title}', '', data['protocol_note'], '',
             f"{totals['tasks']} 题，{totals['task_rounds']} 个任务轮次，{totals['valid_scores']} 个有效分数。主结果为第三轮。", '',
             '| 轮次 | A | B | C | 实际选择 |', '|---|---:|---:|---:|---:|']
    for n, values in metrics.items():
        lines.append('| ' + n + ' | ' + ' | '.join(f'{v:.2f}' for v in values.values()) + ' |')
    lines += ['', '[逐轮结果](rounds.csv) · [第三轮结果](scores.csv) · [完整数据](results.json)', '',
              '实际配置、代码版本和证据以各任务记录为准。官方评分不参与运行时选择。',
              '公开副本已规范路径并脱敏；原始身份哈希与公开文件校验和分开记录。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')

    def link(name, label):
        path = evidence.resolve(name)
        relative = quote(os.path.relpath(path, output), safe='/')
        return f'<a href="{relative}">{html.escape(label)}</a>'

    body = []
    for row in rows:
        links = []
        for arm in 'ABC':
            for field, label in (('receipt', '评分'), ('actor', '执行记录'), ('trace', '轨迹'), ('delivery', '交付物')):
                name = row['evidence'][arm].get(field)
                if name:
                    links.append(link(name, f'{arm} {label}'))
        for key, label in (('checkpoint', 'Checkpoint'), ('pack', 'Reward Pack'), ('configuration', '配置')):
            if row.get(key):
                links.append(link(row[key], label))
        content = ''.join(f'<td>{row[a]:.2f}</td>' for a in 'ABC')
        detail = data.get('task_details', {}).get(row['task'], {})
        expanded = ''.join(f'<p><strong>{html.escape(k)}</strong>: {html.escape(str(v))}</p>' for k,v in detail.items())
        detail_html = f'<details><summary>任务与方法说明</summary>{expanded}</details>' if expanded else ''
        body.append(f'<tr data-round="{row["round"]}"><td>{html.escape(row["task"])}<br>{html.escape(row["name"])}{detail_html}</td>'
                    f'<td>{row["round"]}</td>{content}<td>{row["selected_arm"]} / {row["selected_score"]:.2f}</td>'
                    f'<td>{" · ".join(links)}</td></tr>')
    summary = ''.join(f'<tr><td>{n}</td>'+''.join(f'<td>{v:.2f}</td>' for v in vals.values())+'</tr>'
                      for n, vals in metrics.items())
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{max-width:1440px;margin:36px auto;padding:0 20px;font:16px/1.6 system-ui;color:#213547;background:#f7f9fb}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #d8e0e6;text-align:left}}td:first-child{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}a{{color:#12639b}}input,select{{font:inherit;margin:10px;padding:8px}}td:last-child{{min-width:260px}}.note{{max-width:1000px}}</style>
<h1>{title}</h1><p class="note">{html.escape(data['protocol_note'])}</p>
<p>{totals['tasks']} 题 · 三轮 · {totals['valid_scores']} 个有效分数 · 主结果：第三轮</p>
<p><a href="rounds.csv">逐轮 CSV</a> · <a href="scores.csv">第三轮 CSV</a> · <a href="results.json">完整数据与配置索引</a></p>
<h2>各轮均分</h2><table><thead><tr><th>轮次</th><th>A</th><th>B</th><th>C</th><th>实际选择</th></tr></thead><tbody>{summary}</tbody></table>
<h2>逐题结果与证据</h2><input id="q" placeholder="搜索任务"><select id="round"><option value="">全部轮次</option><option>1</option><option>2</option><option selected>3</option></select>
<div class="scroll"><table><thead><tr><th>任务</th><th>轮次</th><th>A</th><th>B</th><th>C</th><th>实际选择</th><th>证据</th></tr></thead><tbody id="tasks">{''.join(body)}</tbody></table></div>
<p class="note">算法轮次和不同配置均保留。文件来自规范化公开副本，原始身份哈希不等于脱敏后的文件校验和。对照分析冻结的候选集合见关联分析说明。</p>
<script>function filter(){{let q=document.getElementById('q').value.toLowerCase(),n=document.getElementById('round').value;for(let r of document.querySelectorAll('#tasks tr'))r.hidden=!(r.innerText.toLowerCase().includes(q)&&(!n||r.dataset.round===n))}}document.getElementById('q').oninput=filter;document.getElementById('round').onchange=filter;filter()</script></html>'''
    (output / 'report.html').write_text(document)
    write_json(output / 'verification.json', {'status': 'passed', **totals,
               'verified_evidence': len(data['required_evidence']), 'input_sha256': sha256(source), 'model_calls': 0})
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.data_root.resolve(), args.output.resolve(), args.experiment)))


if __name__ == '__main__':
    main()
