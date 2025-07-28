import hashlib
import json
import os
import re
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import Any

from datasets import load_dataset, load_from_disk
from git import InvalidGitRepositoryError, Repo
from gitlab import Gitlab, GitlabGetError 

GITLAB_ISSUE_URL_PATTERN = re.compile(r"gitlab\.[\w.-]+/([^/]+)/([^/]+)(?:/-)?/issues/(\d+)")
GITLAB_REPO_URL_PATTERN = re.compile(r".*[/@]?gitlab(?:\.[a-zA-Z0-9-]+)+\/([^/]+)\/([^/]+)")

CTF_CHALLENGES_CATEGORIES = {
    "rev": "reverse engineering",
    "pwn": "binary exploitation",
    "web": "web security",
    "crypto": "cryptography",
    "misc": "miscellaneous",
    "forensics": "forensics",
}

class NoOutputTimeoutError(TimeoutError):
    """Raised when no output is produced within the specified timeout duration."""
    pass

class InvalidGitlabURL(ValueError):
    """Raised when a GitLab URL is invalid."""
    pass

def get_data_path_name(data_path: str) -> str:
    """If data_path is a file, return the file stem.
    If it's a GitHub or GitLab URL, return the owner__repo_name.
    """
    if data_path.startswith("text://"):
        return hashlib.sha256(data_path.removeprefix("text://").encode()).hexdigest()[:6]
    match = GITHUB_ISSUE_URL_PATTERN.search(data_path)
    if match:
        owner, repo, _ = match.groups()
        return f"{owner}__{repo}"
    match = GITLAB_ISSUE_URL_PATTERN.search(data_path)
    if match:
        owner, repo, _ = match.groups()
        return f"{owner}__{repo}"
    return Path(data_path).stem

def is_gitlab_issue_url(data_path: str) -> bool:
    """Check if data_path is a URL pointing to a GitLab issue."""
    return GITLAB_ISSUE_URL_PATTERN.search(data_path) is not None

def is_gitlab_repo_url(data_path: str) -> bool:
    """Check if data_path is a URL pointing to a GitLab repository.
    Paths to issues or merge requests will also match this pattern.
    """
    return GITLAB_REPO_URL_PATTERN.search(data_path) is not None

def parse_gitlab_issue_url(issue_url: str) -> tuple[str, str, str]:
    """
    Parse a GitLab issue URL and extract the owner, repo, and issue number.

    Returns:
        owner: Repo owner/group.
        repo: Repo name.
        issue_number: Issue number as string.

    Raises:
        InvalidGitlabURL: If the URL is not a valid GitLab issue URL.
    """
    match = GITLAB_ISSUE_URL_PATTERN.search(issue_url)
    if not match:
        msg = f"Invalid GitLab issue URL: {issue_url}"
        raise InvalidGitlabURL(msg)
    res = match.groups()
    assert len(res) == 3
    return tuple(res)  # type: ignore

def parse_gitlab_repo_url(repo_url: str) -> tuple[str, str]:
    """
    Parse a GitLab repository URL and extract the owner and repo name.

    Returns:
        owner: Repo owner/group.
        repo: Repo name.

    Raises:
        InvalidGitlabURL: If the URL is not a valid GitLab repo URL.
    """
    match = GITLAB_REPO_URL_PATTERN.search(repo_url)
    if not match:
        msg = f"Invalid GitLab repository URL: {repo_url}"
        raise InvalidGitlabURL(msg)
    res = match.groups()
    assert len(res) == 2
    return tuple(res)  # type: ignore


def get_gitlab_issue_data(issue_url: str, *, token: str = ""):
    """Returns GitLab issue data in the form of a dictionary.
    See https://docs.gitlab.com/ee/api/issues.html#get-single-issue
    for return format.

    Args:
        issue_url (str): URL of the GitLab issue.
        token (str, optional): GitLab API token.

    Returns:
        Issue data as a dictionary.
    """
    owner, repo, issue_number = parse_gitlab_issue_url(issue_url)
    gl = Gitlab("https://gitlab.com/", private_token=token)
    project = gl.projects.get(f"{owner}/{repo}")
    issue = project.issues.get(int(issue_number))
    return issue.attributes  # Return as a dictionary


class InstanceBuilder:
    def __init__(self, token: str | None = None):
        """This helper class is used to build the data for an instance object,
        retrieving problem statements from GitHub/GitLab issues or local files and setting
        repo paths from GitHub/GitLab URLs or local paths.
        """
        # Args that will be passed to the Instance constructor
        self.args: dict[str, Any] = {}
        self.token = token
        self._instance_id_problem_suffix = ""

    def set_problem_statement_from_gitlab_issue(self, issue_url: str):
        # Fetch raw issue JSON to include creation date
        owner, repo, issue_number = parse_gitlab_issue_url(issue_url)
        issue = get_gitlab_issue_data(issue_url, token=self.token)
        title = issue.get("title", "")
        description = issue.get("description", "")
        created_at = issue.get("created_at") or issue.get("created_on")
        if not created_at:
            raise KeyError(f"GitLab issue JSON missing creation timestamp: {issue_url}")

        self.args["problem_statement"] = f"{title}\n{description}"  # preserve formatting
        self.args["created_at"] = created_at
        self.args["instance_id"] = f"{owner}__{repo}-i{issue_number}"
        self.args["problem_statement_source"] = "online"

    def set_server_description(self, server_name: str | None, port: int | None) -> None:
        """For CTF challenges."""
        if server_name is None or port is None:
            self.args["challenge"]["server_description"] = ""
            return
        if self.args["challenge"]["category"] in {"web", "misc"} and self.args["challenge"].get("proto") != "nc":
            self.args["challenge"]["server_description"] = (
                f"The challenge web server is running on `{server_name}` port `{port}` and you can access it from within the container environment using `curl http://{server_name}:{port}`."
            )
        else:
            self.args["challenge"]["server_description"] = (
                f"The challenge web server is running on `{server_name}` port `{port}` and you can access it from within the container environment using `connect_start {server_name} {port}`."
            )

    def set_problem_statement_from_challenge_json(self, file_path: str) -> None:
        """For CTF challenges."""
        challenge = json.loads(Path(file_path).read_text())
        self.args["challenge"] = challenge
        self.args["challenge"]["files"] = challenge.get("files", [])
        self.args["challenge"]["points"] = challenge.get("points", 10)
        self.args["challenge"]["category_friendly"] = CTF_CHALLENGES_CATEGORIES.get(challenge["category"])
        if (Path(file_path).parent / "docker-compose.yml").is_file():
            logger.debug(f"Found docker_compose file in {Path(file_path).parent}")
            self.args["challenge"]["docker_compose"] = Path(file_path).parent / "docker-compose.yml"
        self.args["challenge"]["port"] = challenge.get("internal_port") or challenge.get("port")
        if "box" in challenge:
            self.args["challenge"]["server_name"] = challenge["box"] or "127.0.0.1"
        else:
            self.args["challenge"]["server_name"] = ""
        self.set_server_description(self.args["challenge"]["server_name"], self.args["challenge"]["port"])
        self.set_problem_statement_from_text(f"{challenge['name']} {challenge['description']}")
        self.args["instance_id"] = (
            # sanitize 'name' to only alphanumeric characters
            challenge.get("category", "misc") + "_" + "".join(a for a in self.args["challenge"]["name"] if a.isalnum())
        )

    def set_problem_statement_from_file(self, file_path: str) -> None:
        if Path(file_path).name == "challenge.json":
            self.set_problem_statement_from_challenge_json(file_path)
        else:
            self.set_problem_statement_from_text(Path(file_path).read_text())

    def set_problem_statement_from_text(self, text: str):
        self.args["problem_statement"] = text
        self.args["instance_id"] = hashlib.sha256(self.args["problem_statement"].encode()).hexdigest()[:6]
        self.args["problem_statement_source"] = "local"

    def set_problem_statement(self, data_path: str):
        """Get problem statement for a single instance from a GitHub/GitLab issue URL or a
        path to a markdown or text file.
        """
        if data_path.startswith("text://"):
            return self.set_problem_statement_from_text(data_path.removeprefix("text://"))
        if is_gitlab_issue_url(data_path):
            return self.set_problem_statement_from_gitlab_issue(data_path)
        if Path(data_path).is_file():
            return self.set_problem_statement_from_file(data_path)
        msg = f"Not sure how to get problem statement from {data_path=}."
        raise ValueError(msg)

    def set_repo_info_from_gitlab_url(self, url: str, base_commit: str | None = None):
        owner, repo = parse_gitlab_repo_url(url)
        self.args["repo"] = f"{owner}/{repo}"
        self.args["repo_type"] = "gitlab"
        # Initialize GitLab client
        gl = Gitlab("https://gitlab.com/", private_token=self.token)
        try:
            project = gl.projects.get(f"{owner}/{repo}")
        except GitlabGetError as e:
            msg = f"Failed to get GitLab project: {e}"
            raise RuntimeError(msg) from e
        if base_commit:
            try:
                commit = project.commits.get(base_commit)
                self.args["base_commit"] = commit.id
            except GitlabGetError as e:
                msg = f"Failed to get GitLab commit {base_commit}: {e}"
                raise RuntimeError(msg) from e
        else:
            try:
                commit = project.commits.list()[0]
                self.args["base_commit"] = commit.id
            except GitlabGetError as e:
                msg = f"Failed to get latest commit from GitLab project: {e}"
                raise RuntimeError(msg) from e
        self.args["version"] = self.args["base_commit"][:7]

    def set_repo_info_from_local_path(self, path: str, base_commit: str | None = None):
        self.args["repo"] = str(Path(path).resolve())
        self.args["repo_type"] = "local"
        if base_commit:
            self.args["base_commit"] = base_commit
        else:
            try:
                repo = Repo(path, search_parent_directories=True)
            except InvalidGitRepositoryError as e:
                msg = f"Could not find git repository at {path=}."
                raise ValueError(msg) from e
            if repo.is_dirty() and "PYTEST_CURRENT_TEST" not in os.environ:
                msg = f"Local git repository {path} is dirty. Please commit or stash changes."
                raise ValueError(msg)
            self.args["base_commit"] = repo.head.object.hexsha
        self.args["version"] = self.args["base_commit"][:7]

    def set_repo_info(self, repo: str, base_commit: str | None = None):
        if is_gitlab_repo_url(repo):
            self.set_repo_info_from_gitlab_url(repo, base_commit=base_commit)
        elif Path(repo).is_dir():
            self.set_repo_info_from_local_path(repo, base_commit=base_commit)
        else:
            msg = f"Could not determine repo path from {repo=}."
            raise ValueError(msg)

    def set_from_dict(self, instance_dict: dict[str, Any]):
        self.args |= instance_dict

    def set_missing_fields(self):
        # TODO: This field is only needed while swe_env is using some questionable logic
        # to determine whether to clone from a mirror or not. This should be removed in the future.
        # Values: 'swe-bench' (loaded from json/jsonl for swe-bench style inference),
        # 'online' (loaded from GitHub/GitLab issue or similar) or 'local' (loaded from local file)
        if "problem_statement_source" not in self.args:
            self.args["problem_statement_source"] = "swe-bench"
        if "repo_type" not in self.args:
            self.args["repo_type"] = "github"

    def validate(self):
        required_fields = [
            "problem_statement",
            "instance_id",
            "repo",
            "repo_type",
            "base_commit",
            "version",
            "problem_statement_source",
        ]
        if not all(x in self.args for x in required_fields):
            missing = set(required_fields) - set(self.args.keys())
            msg = f"Missing required fields: {missing=}"
            raise ValueError(msg)
        if self.args["repo_type"] not in {"github", "gitlab", "local"}:
            msg = f"Invalid repo type: {self.args['repo_type']=}"
            raise ValueError(msg)
        if self.args["repo_type"] == "github" and self.args["repo"].count("/") != 1:
            msg = f"Invalid repo format for {self.args['repo_type']=}: {self.args['repo']=}"
            raise ValueError(msg)
        if self.args["repo_type"] == "gitlab" and self.args["repo"].count("/") != 1:
            msg = f"Invalid repo format for {self.args['repo_type']=}: {self.args['repo']=}"
            raise ValueError(msg)

    def build(self) -> dict[str, Any]:
        self.set_missing_fields()
        self.validate()
        return self.args

def get_instances(
    file_path: str,
    base_commit: str | None = None,
    split: str | None = None,
    token: str | None = None,
    *,
    repo_path: str = "",
) -> list[dict[str, Any]]:
    """
    Getter function for handling JSON, JSONL files, GitHub/GitLab issues, or datasets.

    Args:
        file_path (str): Path to file or URL.
        base_commit (str, optional): Base commit hash or branch/tag name.
        split (str, optional): Dataset split.
        token (str, optional): API token for GitHub/GitLab.
        repo_path (str, optional): Path to local repository or GitHub/GitLab repo URL.

    Returns:
        List of instances as dictionaries.
    """

    def instance_from_dict(instances: dict[str, Any]) -> dict[str, Any]:
        ib = InstanceBuilder(token=token)
        ib.set_from_dict(instances)
        return ib.build()

    def postproc_instance_list(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [instance_from_dict(x) for x in instances]

    # The next if statements are very brittle logic to determine if we're processing a single instance
    if (
        file_path.startswith("text://")
        or (
            Path(file_path).is_file()
            and (Path(file_path).suffix in [".md", ".txt"] or Path(file_path).name == "challenge.json")
        )
        or is_gitlab_issue_url(file_path)
    ):
        ib = InstanceBuilder(token=token)
        ib.set_problem_statement(file_path)
        if repo_path:
            ib.set_repo_info(repo_path, base_commit=base_commit)
        else:
            if is_gitlab_repo_url(file_path):
                ib.set_repo_info_from_gitlab_url(file_path, base_commit=base_commit)
            elif Path(file_path).is_dir():
                ib.set_repo_info_from_local_path(file_path, base_commit=base_commit)
            else:
                msg = f"Could not determine repo path from {file_path=}, {repo_path=}"
                raise ValueError(msg)
        return [ib.build()]

    if base_commit:
        msg = "base_commit must be empty if running over multiple problem statements"
        raise ValueError(msg)

    if repo_path:
        if not Path(repo_path).exists():
            msg = f"Specified repository path {repo_path} does not exist"
            raise FileNotFoundError(msg)
        msg = "repo_path must be empty if running over multiple problem statements"
        raise ValueError(msg)

    # If file_path is a directory, attempt load from disk
    if Path(file_path).is_dir():
        try:
            dataset_or_dict = load_from_disk(file_path)
            if isinstance(dataset_or_dict, dict):
                return postproc_instance_list(dataset_or_dict[split])
            return postproc_instance_list(dataset_or_dict)
        except FileNotFoundError:
            # Raised by load_from_disk if the directory is not a dataset directory
            pass

    if base_commit is not None:
        msg = "base_commit must be None if data_path is not a GitHub/GitLab issue URL"
        raise ValueError(msg)

    # If file_path is a file, load the file
    if file_path.endswith(".json"):
        with open(file_path) as file:
            return postproc_instance_list(json.load(file))
    if file_path.endswith(".jsonl"):
        return postproc_instance_list([json.loads(x) for x in Path(file_path).read_text().splitlines(keepends=True)])

    # Attempt load from HF datasets as a last resort
    try:
        return postproc_instance_list(load_dataset(file_path, split=split))
    except Exception as e:
        msg = (
            f"Could not load instances from {file_path}. "
            "Please ensure --data_path is a GitHub/GitLab URL, a SWE-bench HuggingFace dataset, or a JSON/JSONL file."
        )
        raise ValueError(msg) from e

def get_associated_commit_urls(org: str, repo: str, issue_number: str, *, token: str = "") -> list[str]:
    """Return the URLs of commits that would close an issue.

    Supports both GitHub and GitLab repositories based on the repo_type.

    Args:
        org (str): Repository owner or group.
        repo (str): Repository name.
        issue_number (str): Issue number as a string.
        token (str, optional): API token for GitHub/GitLab.

    Returns:
        List of commit URLs that reference the issue for closing.
    """
    # Determine if the repository is GitHub or GitLab
    # This function assumes that the repo_type is known or can be inferred.
    # For this implementation, let's assume GitHub. Modify as needed.
    # Alternatively, you could pass an additional parameter indicating the platform.
    # For demonstration, we'll handle both based on token presence.
    commit_urls = []
    if "gitlab.com" in repo:
        try:
            issue_url = f"https://gitlab.com//{org}/{repo}/-/issues/{issue_number}"
            if is_gitlab_issue_url(issue_url):
                gl = Gitlab("https://gitlab.com/", private_token=token)
                project = gl.projects.get(f"{org}/{repo}")
                notes = project.issues.list_notes(issue=int(issue_number))
                for note in notes:
                    if note.note.startswith("Closes"):
                        match = re.search(r"closes\s+#(\d+)", note.note, re.IGNORECASE)
                        if match:
                            # Assuming the commit is linked via reference in the note
                            # GitLab might not provide direct commit references in issue notes
                            # You might need to parse commit messages or references differently
                            # This is a placeholder for actual implementation
                            # For example, fetch all commits and check their messages
                            commits = project.commits.list()
                            for commit in commits:
                                if (
                                    f"closes #{issue_number}" in commit.message.lower()
                                    or f"fixes #{issue_number}" in commit.message.lower()
                                ):
                                    commit_urls.append(commit.web_url)
        except Exception as e:
            logger.error(f"Error fetching associated GitLab commit URLs: {e}", exc_info=True)
    else:
        logger.warning(f"Repository {org}/{repo} is neither GitHub nor GitLab based on the provided information.")
    return commit_urls
