"""Single-agent capability contract shared by schema exposure and permissions."""

from copy import deepcopy

from reward_engine.adapters.workbuddy.io import digest

MODES = ("agent.code.normal", "agent.work.normal")
OFFICE_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "list_files",
    "grep",
    "bash",
    "todo_create",
    "todo_list",
    "todo_get",
    "todo_modify",
)
MANAGEMENT_TOOLS = (
    "subagent_spawn",
    "subagent_send_input",
    "subagent_resume",
    "subagent_wait",
    "subagent_list",
    "subagent_close",
    "task_tool",
    "switch_mode",
    "cron",
    "cron_create_job",
    "cron_update_job",
    "cron_toggle_job",
    "cron_list_jobs",
    "cron_get_job",
    "cron_delete_job",
    "cron_preview_job",
    "submit_goal_report",
    "get_current_goal",
    "user_todos",
)
BUILTIN_TOOLS = (
    *OFFICE_TOOLS,
    *MANAGEMENT_TOOLS,
    "powershell",
    "mcp_exec_command",
    "create_terminal",
    "tool_search",
    "tool_call",
    "lsp",
)


def mode(config):
    value = config.get("jiuwenswarm_mode", MODES[0])
    if value not in MODES:
        raise ValueError(f"Unsupported jiuwenswarm_mode: {value}; expected {MODES}")
    return value


def policy(allowed_tools=None):
    names = sorted(set(OFFICE_TOOLS if allowed_tools is None else allowed_tools))
    if allowed_tools is not None and any(not n.startswith("mcp_claw_") for n in names):
        raise ValueError("Claw tools must be exact registered mcp_claw_* names")
    return {
        "version": "jiuwenswarm-task-tools-v1",
        "kind": "office" if allowed_tools is None else "claw",
        "allowed": names,
        "required": ["bash", "read_file", "write_file"]
        if allowed_tools is None
        else names,
    }


def permissions(contract, *, sandboxed_shell=False):
    result = {
        "enabled": True,
        "schema": "tiered_policy",
        "mode": "manual",
        "permission_mode": "normal",
        "package_builtin_rules": False,
        "defaults": {"*": "deny"},
        "tools": {n: "deny" for n in BUILTIN_TOOLS},
        "rules": [],
    }
    result["tools"].update({n: "allow" for n in contract["allowed"]})
    if sandboxed_shell:
        result["shell_guard"] = {"unknown_structure": False, "interpreter_sink": False}
        result["external_directory"] = {"*": "allow"}
    return result


def filter_request(body, contract):
    filtered = deepcopy(body)
    original = [t.get("function", {}).get("name") for t in body.get("tools", [])]
    if any(not isinstance(n, str) for n in original) or len(original) != len(
        set(original)
    ):
        raise ValueError("Invalid or duplicate actor tool names")
    missing = sorted(set(contract["required"]) - set(original))
    if missing:
        raise ValueError(
            "Required actor tools are not registered: " + ", ".join(missing)
        )
    selected = [
        t
        for t in filtered.get("tools", [])
        if t["function"]["name"] in contract["allowed"]
    ]
    sent = [t["function"]["name"] for t in selected]
    choice = filtered.get("tool_choice")
    if (
        choice is not None
        and not isinstance(choice, dict)
        and choice not in ("auto", "none", "required")
    ):
        raise ValueError("Invalid actor tool_choice")
    if isinstance(choice, dict) and (
        set(choice) != {"type", "function"}
        or choice.get("type") != "function"
        or not isinstance(choice.get("function"), dict)
        or set(choice["function"]) != {"name"}
    ):
        raise ValueError("Invalid actor tool_choice")
    if isinstance(choice, dict) and choice.get("function", {}).get("name") not in sent:
        raise ValueError(
            "tool_choice specifies an unavailable actor tool: " + str(choice)
        )
    if choice == "required" and not selected:
        raise ValueError(
            "tool_choice requires a tool but the actor policy exposes none"
        )
    filtered["tools"] = selected
    return filtered, {
        "original_tools": original,
        "sent_tools": sent,
        "policy_sha256": digest(contract),
    }


def capabilities(config, contract):
    return {
        "mode": mode(config),
        "tool_policy": contract,
        "tool_policy_sha256": digest(contract),
        "enabled": ["single_agent", "task_tools", "same_session_resume"],
        "disabled": [
            "team",
            "subagents",
            "long_term_memory",
            "network_skills",
            "cron",
            "global_goals",
        ],
    }
