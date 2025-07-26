import debugpy
debugpy.listen(("0.0.0.0", 5678))  # Listen on all interfaces, port 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # Pause until debugger attaches

import os
import subprocess
import sys
from gitlab_utils import get_instances

def get_instance_gitlab_issue():
    token = os.environ.get("OP_TOKEN")
    instance = get_instances(
        "https://gitlab.com/open-darts/open-darts/-/issues/1",
        token=token
    )[0]
    compare_with = {
        "repo": "open-darts/open-darts",
        "instance_id": "open-darts__open-darts-i1",
        "repo_type": "gitlab",
    }
    for key in compare_with:
        assert instance[key] == compare_with[key]
    assert "problem_statement" in instance
    assert len(instance["base_commit"]) > 10
    assert instance["version"]


def setup_codex_codebase(repo_info: dict, repo_path: str) -> str:
    """
    Clone or prepare the codebase for Codex CLI, ensuring dependencies are installed
    and the repo is at the correct base commit.
    """
    target_path = repo_path

    # Clone or update GitLab repo
    if repo_info["repo_type"] == "gitlab":
        clone_url = f"https://gitlab.com/{repo_info['repo']}.git"
        if os.path.isdir(target_path):
            print(f"Updating existing repo at {target_path}")
            subprocess.run(["git", "-C", target_path, "fetch"], check=True)
            subprocess.run(["git", "-C", target_path, "checkout", repo_info["base_commit"]], check=True)
            subprocess.run(["git", "-C", target_path, "reset", "--hard", repo_info["base_commit"]], check=True)
        else:
            print(f"Cloning {clone_url} into {target_path}")
            subprocess.run(["git", "clone", clone_url, target_path], check=True)
            subprocess.run(["git", "-C", target_path, "checkout", repo_info["base_commit"]], check=True)

    elif repo_info["repo_type"] == "local":
        target_path = repo_info["repo"]

    else:
        raise ValueError(f"Unsupported repo_type: {repo_info['repo_type']}")

    # Install Python dependencies if requirements.txt exists
    req_file = os.path.join(target_path, "requirements.txt")
    if os.path.isfile(req_file):
        print("Installing Python dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)

    # Install Node.js dependencies if package.json exists
    pkg_file = os.path.join(target_path, "package.json")
    if os.path.isfile(pkg_file):
        print("Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=target_path, check=True)

    # Ensure OPENAI_API_KEY is set for Codex CLI
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set for Codex CLI")

    # Configure Codex CLI to use this codebase
    print("Configuring Codex CLI for repository...")
    subprocess.run(["codex", "config", "set", "repo", target_path], check=True)

    return target_path


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
    # Retrieve issue instance from GitLab
    issue_instance = get_instance_gitlab_issue()

    # Determine local repository path (override with REPO_PATH env var if set)
    repo_path = os.environ.get("REPO_PATH", "/tmp/repo")

    # Setup the codebase for the Codex CLI
    prepared_path = setup_codex_codebase(issue_instance, repo_path)

    # Run Codex CLI to generate a proposed solution
    run_codex_on_issue(issue_instance, prepared_path)