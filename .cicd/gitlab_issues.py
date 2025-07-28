import os
import re
import subprocess
import sys
import git
from datetime import datetime, timezone
from gitlab_utils import get_instances


def get_instance_gitlab_issue():
    token = os.environ.get("OP_TOKEN")
    instance = get_instances(
        "https://gitlab.com/open-darts/open-darts/-/issues/82",
        token=token
    )[0]
    assert instance.get("repo"), "Missing repository info"
    assert instance.get("instance_id"), "Missing issue instance ID"
    assert "problem_statement" in instance, "Missing problem statement"
    assert "created_at" in instance, "Missing issue creation date"
    return instance


def extract_commit_hash_via_llm(issue_text: str) -> str | None:
    """
    Ask Codex CLI (LLM) to extract a commit hash from the issue text.
    Returns the hash string or None if not found.
    """
    prompt = (
        "Extract the Git commit hash (7–40 hex characters) from the following issue text. "
        "If no commit hash is mentioned, respond with NONE.\n---\n" + issue_text
    )
    # Pass prompt as positional arg per Codex CLI usage
    result = subprocess.run(
        ["codex", "exec", prompt],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        print(f"Warning: Codex CLI returned status {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"Codex stderr: {result.stderr}", file=sys.stderr)
    # Look for "Agent message:" and extract following text
    agent_msg = None
    for line in output.splitlines()[::-1]:  # scan backwards for last occurrence
        if "Agent message:" in line:
            parts = line.split("Agent message:", 1)
            agent_msg = parts[1].strip()
            break
    text_to_search = agent_msg if agent_msg is not None else output
    if agent_msg is None:
        print("Warning: 'Agent message:' not found in LLM output, using full output", file=sys.stderr)
    # Extract a valid hash from the extracted text
    match = re.search(r"\b[0-9a-f]{7,40}\b", text_to_search)
    return match.group(0) if match else None

def find_closest_commit_on_branch(repo: git.Repo, branch: str, issue_date: str) -> str:
    """
    Find the commit on `branch` whose commit date is closest to issue_date.
    issue_date must be an ISO 8601 string.
    """
    target_dt = datetime.fromisoformat(issue_date.replace("Z", "+00:00")).astimezone(timezone.utc)
    commits = list(repo.iter_commits(branch))
    closest = None
    min_diff = None
    for commit in commits:
        dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
        diff = abs((dt - target_dt).total_seconds())
        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest = commit.hexsha
    if not closest:
        raise RuntimeError(f"No commits found on branch {branch}")
    return closest


def prepare_repository_state(issue: dict) -> str:
    """
    Checkout the commit mentioned in the issue via LLM extraction or fallback to `development` branch.
    Configure Codex CLI to use this path.
    """
    repo_path = os.getcwd()
    repo = git.Repo(repo_path)
    problem = issue["problem_statement"]
    created_at = issue["created_at"]

    # Try LLM extraction
    commit_hash = extract_commit_hash_via_llm(problem)
    if commit_hash:
        print(f"LLM-extracted commit hash: {commit_hash}")
    else:
        print("Warning: Could not extract commit hash via LLM. Falling back to development branch closest commit.")
        commit_hash = find_closest_commit_on_branch(repo, "development", created_at)
        print(f"Using fallback commit {commit_hash} from development branch")

    # Checkout the chosen commit
    print(f"Checking out commit {commit_hash} in {repo_path}")
    repo.git.checkout(commit_hash)

    # Verify working tree is clean
    if repo.is_dirty(untracked_files=True):
        print("Warning: working tree not clean after checkout")

    # Ensure OPENAI_API_KEY is set for Codex CLI
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set for Codex CLI"
        )

    # Configure Codex CLI to point to this codebase
    print("Configuring Codex CLI to use this codebase...")
    subprocess.run(
        ["codex", "config", "set", "repo", repo_path],
        check=True
    )
    return repo_path


def run_codex_on_issue(issue: dict, repo_path: str):
    """
    Invoke Codex CLI to propose a solution for the given issue.
    """
    issue_id = issue["instance_id"]
    print(f"Running Codex CLI to solve issue {issue_id}...")
    cmd = [
        "codex", "solve",
        "--repo", repo_path,
        "--issue-id", issue_id,
        "--output", f"{issue_id}_proposal.diff"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Codex CLI error output:", result.stderr)


if __name__ == "__main__":
    issue_instance = get_instance_gitlab_issue()
    repo_path = prepare_repository_state(issue_instance)
    run_codex_on_issue(issue_instance, repo_path)