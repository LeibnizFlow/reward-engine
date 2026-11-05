"""Build hash-locked Office overlays and exercise the offline task environment."""

import argparse
import json
import subprocess
from pathlib import Path
from uuid import uuid4

from reward_engine.adapters.workbuddy.io import file_hash, read, write

PINS = [
    "python-pptx==1.0.2",
    "lxml==6.1.3",
    "Pillow==12.3.0",
    "XlsxWriter==3.2.9",
    "typing-extensions==4.16.0",
    "openpyxl==3.1.5",
    "et-xmlfile==2.0.0",
]
PROBE = r"""
import hashlib, importlib.metadata as m, json, pathlib, sqlite3, ssl, tempfile
import pip, pptx, lxml.etree, PIL, xlsxwriter, typing_extensions, openpyxl
root = pathlib.Path('/workspace')
inputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*')) if p.is_file()}
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    deck = pptx.Presentation(); deck.slides.add_slide(deck.slide_layouts[6]); deck.save(d/'a.pptx')
    deck = pptx.Presentation(d/'a.pptx'); deck.slides.add_slide(deck.slide_layouts[6]); deck.save(d/'b.pptx')
    assert len(pptx.Presentation(d/'b.pptx').slides) == 2
    with xlsxwriter.Workbook(d/'a.xlsx') as book: book.add_worksheet().write(0, 0, 'before')
    book = openpyxl.load_workbook(d/'a.xlsx'); book.active['A1']='after'; book.save(d/'b.xlsx')
    assert openpyxl.load_workbook(d/'b.xlsx').active['A1'].value == 'after'
with tempfile.TemporaryDirectory(dir=root) as d:
    p=pathlib.Path(d)/'delivery'; p.write_text('writable'); assert p.read_text() == 'writable'
print(json.dumps({'inputs': inputs, 'versions': {n:m.version(n) for n in
 ['python-pptx','lxml','Pillow','XlsxWriter','typing-extensions','openpyxl']},
 'ssl':ssl.OPENSSL_VERSION,'sqlite':sqlite3.sqlite_version,'offline_roundtrip':True}))
"""


def run(*args):
    result = subprocess.run(
        list(map(str, args)), capture_output=True, text=True, timeout=600
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])
    return result.stdout


def input_hashes(image):
    script = "import pathlib,hashlib,json; r=pathlib.Path('/workspace'); print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('*')) if p.is_file()}))"
    return json.loads(
        run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            image,
            "-c",
            script,
        )
    )


def probe_image(image):
    # Match task startup: root initializes inside the isolated container, then
    # all task Python/file operations run as dev with no network or extra mounts.
    name = "jiuwen-office-preflight-" + uuid4().hex
    run(
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--user",
        "root",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "--entrypoint",
        "sleep",
        image,
        "infinity",
    )
    try:
        run(
            "docker",
            "exec",
            "--user",
            "root",
            name,
            "bash",
            "-lc",
            "id dev >/dev/null 2>&1 || useradd -m -s /bin/bash dev; "
            "mkdir -p /workspace; chown -R dev:dev /workspace",
        )
        return json.loads(
            run("docker", "exec", "--user", "dev", name, "python", "-c", PROBE)
        )
    finally:
        run("docker", "rm", "-f", name)


def build(source_manifest, target, tasks):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    records = read(source_manifest)
    chosen = [r for r in records["records"] if r["task_id"] in tasks]
    if {r["task_id"] for r in chosen} != set(tasks):
        raise ValueError("Source manifest lacks requested task images")
    wheels = target / "wheels"
    wheels.mkdir()
    # Download with the exact base interpreter/ABI, never install at task runtime.
    run(
        "docker",
        "run",
        "--rm",
        "--user",
        "root",
        "-v",
        f"{wheels}:/wheels",
        "--entrypoint",
        "python",
        chosen[0]["image_id"],
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--no-deps",
        "--dest",
        "/wheels",
        *PINS,
    )
    hashes = {p.name: file_hash(p) for p in wheels.iterdir()}
    lock = []
    for pin in PINS:
        package = pin.split("==")[0].lower().replace("-", "_")
        matches = [
            p for p in wheels.iterdir() if p.name.lower().startswith(package + "-")
        ]
        if len(matches) != 1:
            raise ValueError("Wheel not uniquely locked: " + pin)
        lock.append(pin + " --hash=sha256:" + hashes[matches[0].name])
    (target / "requirements.lock").write_text("\n".join(lock) + "\n")
    for row in chosen:
        base = row["image_id"]
        original = input_hashes(base)
        (target / "Dockerfile").write_text(
            f"FROM {base}\nUSER root\nCOPY wheels /opt/office-wheels\nCOPY requirements.lock /opt/office-wheels/requirements.lock\nRUN python -m pip install --no-index --no-deps --require-hashes --find-links=/opt/office-wheels -r /opt/office-wheels/requirements.lock && rm -rf /opt/office-wheels\nRUN (id dev >/dev/null 2>&1 || useradd -m -s /bin/bash dev) && chown -R dev:dev /workspace\nUSER dev\nWORKDIR /workspace\n"
        )
        tag = "jiuwenswarm-office:" + file_hash(target / "Dockerfile")[:24]
        run("docker", "build", "--network", "none", "-t", tag, target)
        image = json.loads(run("docker", "image", "inspect", tag))[0]["Id"]
        proof = probe_image(image)
        if proof["inputs"] != original:
            raise ValueError("Derived image changed task inputs")
        row.update(
            base_image_id=base,
            image_id=image,
            derived_tag=tag,
            input_sha256=original,
            dependency_versions=proof["versions"],
        )
        write(target / (row["task_id"] + ".probe.json"), proof)
    records = {"records": chosen}
    records.update(
        office_overlay={
            "wheels": hashes,
            "pins": PINS,
            "source_manifest_sha256": file_hash(source_manifest),
        }
    )
    write(target / "manifest.json", records)
    return target / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    print(build(args.source_manifest, args.output, args.tasks))


def configure_images(config, tasks_root):
    """Bind each prepared task to a locally available immutable offline image."""
    records = {r["task_id"]: r for r in read(config["task_images_manifest"])["records"]}
    for task in config["task_ids"]:
        row = records[task]
        observed = json.loads(run("docker", "image", "inspect", row["image_id"]))[0][
            "Id"
        ]
        if observed != row["image_id"]:
            raise ValueError("Office image identity changed")
        proof = probe_image(observed)
        if proof["inputs"] != row["input_sha256"]:
            raise ValueError("Office image input hashes changed")
        # WorkBuddy uses a task-owned Compose service. Keep its mounts, network,
        # user and verifier settings, replacing only build/image selection.
        import yaml

        path = Path(tasks_root) / task / "environment/docker-compose.yaml"
        if not path.exists():
            path = path.with_suffix(".yml")
        document = yaml.safe_load(path.read_text())
        main = document["services"]["main"]
        main.pop("build", None)
        main["image"] = observed
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        # Harbor selects its prebuilt Compose layer from task.toml, not from
        # the task's Compose image alone. Bind both to prevent an implicit build.
        import toml

        task_path = Path(tasks_root) / task / "task.toml"
        task_config = toml.loads(task_path.read_text())
        task_config.setdefault("environment", {})["docker_image"] = observed
        task_path.write_text(toml.dumps(task_config))


def freeze_images(config, tasks_root, output):
    """Resolve default task builds to immutable local IDs before freezing a run."""
    records = []
    from .process import tree_hash

    for task in config["task_ids"]:
        environment = Path(tasks_root) / task / "environment"
        tag = "reward-engine-office:" + tree_hash(environment)[:24]
        run("docker", "build", "-t", tag, environment)
        image = json.loads(run("docker", "image", "inspect", tag))[0]["Id"]
        proof = probe_image(image)
        records.append(
            {
                "task_id": task,
                "image_id": image,
                "input_sha256": proof["inputs"],
                "dependency_versions": proof["versions"],
            }
        )
    target = Path(output) / "task_images.json"
    write(target, {"format": "reward-engine-task-images-v1", "records": records})
    config.update(
        task_images_manifest=str(target), task_images_sha256=file_hash(target)
    )
    return target
